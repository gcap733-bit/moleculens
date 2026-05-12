# ==========================================
# fetcher.py — data fetching with retries + disk caching
# Sources: ChEMBL, PubChem, pkCSM
# ==========================================

import os
import math
import time
import json
import hashlib
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import Descriptors
from chembl_webresource_client.new_client import new_client

from config import (
    MAX_DRUGS, PUBCHEM_DELAY, PKCMS_DELAY,
    MAX_RETRIES, RETRY_BACKOFF, CACHE_DIR, CACHE_ENABLED
)

os.makedirs(CACHE_DIR, exist_ok=True)


# ==========================================
# CACHING HELPERS
# ==========================================
def _cache_key(prefix: str, identifier: str) -> str:
    h = hashlib.md5(identifier.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{prefix}_{h}.json")


def _load_cache(path: str):
    if CACHE_ENABLED and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save_cache(path: str, data):
    if CACHE_ENABLED:
        with open(path, "w") as f:
            json.dump(data, f)


# ==========================================
# RETRY WRAPPER
# ==========================================
def _with_retry(fn, *args, **kwargs):
    """
    Calls fn(*args, **kwargs) up to MAX_RETRIES times with
    exponential backoff. Returns None on total failure.
    """
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"    [!] Failed after {MAX_RETRIES} attempts: {e}")
                return None
            print(f"    [!] Attempt {attempt} failed ({e}), retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay *= RETRY_BACKOFF


# ==========================================
# CHEMBL — drug discovery + base properties
# ==========================================
def fetch_chembl_drugs(disease_name: str, max_drugs: int = MAX_DRUGS) -> pd.DataFrame:
    """
    Fetches drugs for a disease from ChEMBL and computes
    RDKit descriptors as baseline physicochemical properties.
    ChEMBL is the only free source for disease-to-drug mapping.
    """
    print(f"[*] ChEMBL: fetching drugs for '{disease_name}'...")

    cache_path = _cache_key("chembl", disease_name)
    cached = _load_cache(cache_path)
    if cached:
        print(f"    [cache] Loaded {len(cached)} drugs from disk.")
        return pd.DataFrame(cached)

    def _fetch():
        indication_api = new_client.drug_indication
        indications = indication_api.filter(
            mesh_heading__icontains=disease_name
        ).only(["molecule_chembl_id"])
        chembl_ids = list(set(ind["molecule_chembl_id"] for ind in indications))
        if not chembl_ids:
            return []
        molecule_api = new_client.molecule
        return list(
            molecule_api.filter(
                molecule_chembl_id__in=chembl_ids[:max_drugs]
            ).only(["molecule_chembl_id", "molecule_structures"])
        )

    molecules = _with_retry(_fetch)
    if not molecules:
        print(f"    [!] No drugs returned from ChEMBL.")
        return pd.DataFrame()

    data = []
    for mol in molecules:
        try:
            smiles = mol["molecule_structures"]["canonical_smiles"]
            rd_mol = Chem.MolFromSmiles(smiles)
            if rd_mol is None:
                continue
            data.append({
                "ChEMBL_ID": mol["molecule_chembl_id"],
                "SMILES": smiles,
                "MolWt":    Descriptors.MolWt(rd_mol),
                "LogP":     Descriptors.MolLogP(rd_mol),
                "TPSA":     Descriptors.TPSA(rd_mol),
                "HBD":      Descriptors.NumHDonors(rd_mol),
                "HBA":      Descriptors.NumHAcceptors(rd_mol),
                "RotBonds": Descriptors.NumRotatableBonds(rd_mol),
                "MolMR":    Descriptors.MolMR(rd_mol),
            })
        except:
            continue

    _save_cache(cache_path, data)
    print(f"    [✓] {len(data)} drugs fetched and cached.")
    return pd.DataFrame(data)


# ==========================================
# PUBCHEM — 20 physicochemical properties
# Preferred over ChEMBL for physicochemical props:
# standardized structures, validated XLogP3, unique
# properties (Complexity, FormalCharge, stereo counts)
# ==========================================
PUBCHEM_PROPS = ",".join([
    "MolecularFormula", "MolecularWeight", "XLogP", "ExactMass",
    "MonoisotopicMass", "TPSA", "Complexity", "Charge",
    "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount",
    "HeavyAtomCount", "IsotopeAtomCount", "AtomStereoCount",
    "DefinedAtomStereoCount", "UndefinedAtomStereoCount",
    "BondStereoCount", "DefinedBondStereoCount",
    "UndefinedBondStereoCount", "CovalentUnitCount",
])


def _fetch_pubchem_single(smiles: str) -> dict:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"property/{PUBCHEM_PROPS}/JSON"
    )
    resp = requests.post(url, data={"smiles": smiles}, timeout=15)
    resp.raise_for_status()
    props = resp.json()["PropertyTable"]["Properties"][0]
    return {
        "PC_CID":                      props.get("CID"),
        "PC_MolecularFormula":         props.get("MolecularFormula"),
        "PC_MolecularWeight":          props.get("MolecularWeight"),
        "PC_XLogP":                    props.get("XLogP"),
        "PC_ExactMass":                props.get("ExactMass"),
        "PC_MonoisotopicMass":         props.get("MonoisotopicMass"),
        "PC_TPSA":                     props.get("TPSA"),
        "PC_Complexity":               props.get("Complexity"),
        "PC_FormalCharge":             props.get("Charge"),
        "PC_HBD":                      props.get("HBondDonorCount"),
        "PC_HBA":                      props.get("HBondAcceptorCount"),
        "PC_RotatableBonds":           props.get("RotatableBondCount"),
        "PC_HeavyAtomCount":           props.get("HeavyAtomCount"),
        "PC_IsotopeAtomCount":         props.get("IsotopeAtomCount"),
        "PC_AtomStereoCount":          props.get("AtomStereoCount"),
        "PC_DefinedAtomStereoCount":   props.get("DefinedAtomStereoCount"),
        "PC_UndefinedAtomStereoCount": props.get("UndefinedAtomStereoCount"),
        "PC_BondStereoCount":          props.get("BondStereoCount"),
        "PC_DefinedBondStereoCount":   props.get("DefinedBondStereoCount"),
        "PC_UndefinedBondStereoCount": props.get("UndefinedBondStereoCount"),
        "PC_CovalentUnitCount":        props.get("CovalentUnitCount"),
    }


def _fetch_pubchem_cached(args):
    """Worker for parallel PubChem fetching. Returns (index, props)."""
    i, smiles = args
    cache_path = _cache_key("pubchem", smiles)
    cached = _load_cache(cache_path)
    if cached:
        return i, cached
    props = _with_retry(_fetch_pubchem_single, smiles) or {}
    _save_cache(cache_path, props)
    time.sleep(PUBCHEM_DELAY)  # Respect rate limit per worker
    return i, props


def enrich_with_pubchem(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches PubChem properties in parallel using ThreadPoolExecutor.
    Uses 5 workers — stays safely within PubChem's 5 req/sec free limit.
    Results are re-ordered to match original df order.
    """
    print("[*] PubChem: fetching physicochemical properties (parallel, 5 workers)...")
    smiles_list = list(df["SMILES"])
    results = [None] * len(smiles_list)
    completed = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_pubchem_cached, (i, s)): i
                   for i, s in enumerate(smiles_list)}
        for future in as_completed(futures):
            i, props = future.result()
            results[i] = props
            completed += 1
            if completed % 10 == 0:
                print(f"    ... {completed}/{len(smiles_list)} done")

    pubchem_df = pd.DataFrame(results)
    print(f"    [✓] PubChem properties added ({len(pubchem_df.columns)} columns).")
    return pd.concat([df.reset_index(drop=True), pubchem_df.reset_index(drop=True)], axis=1)


# ==========================================
# pkCSM — ADMET pharmacokinetic properties
# Only free REST API with full ADMET predictions
# from SMILES. No API key required.
# ==========================================
def _fetch_pkcms_single(smiles: str) -> dict:
    """
    pkCSM ADMET prediction — tries multiple endpoints and request formats.
    pkCSM has historically changed URLs and accepted formats.
    We try every known combination so at least one works.
    Strategies:
      1. biosig.unimelb.edu.au — form-data POST
      2. biosig.unimelb.edu.au — JSON POST
      3. biosig.lab.uq.edu.au  — form-data POST (old URL)
      4. biosig.lab.uq.edu.au  — JSON POST (old URL)
    If all fail, compute RDKit-based ADMET approximations as fallback.
    """
    endpoints = [
        ("https://biosig.unimelb.edu.au/pkcsm/api/v1/prediction", "form"),
        ("https://biosig.unimelb.edu.au/pkcsm/api/v1/prediction", "json"),
        ("https://biosig.lab.uq.edu.au/pkcsm/api/v1/prediction",  "form"),
        ("https://biosig.lab.uq.edu.au/pkcsm/api/v1/prediction",  "json"),
    ]
    for url, fmt in endpoints:
        try:
            if fmt == "form":
                resp = requests.post(url, data={"smiles": smiles}, timeout=30)
            else:
                resp = requests.post(
                    url,
                    json={"smiles": smiles},
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
            if resp.status_code == 200:
                try:
                    result = resp.json()
                    if isinstance(result, dict) and len(result) > 1:
                        return {
                            f"ADMET_{k}": v
                            for k, v in result.items()
                            if k.lower() not in ("smiles", "error")
                        }
                except Exception:
                    continue
        except Exception:
            continue

    # ── Fallback: RDKit-based ADMET approximations ─────────────────────
    # When pkCSM is unavailable, compute key ADMET properties locally
    # using well-established RDKit descriptors and Lipinski-based rules.
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        arom  = rdMolDescriptors.CalcNumAromaticRings(mol)

        # Absorption approximations (rule-based)
        gi_absorption = "High" if (mw < 500 and logp < 5 and hbd <= 5 and hba <= 10) else "Low"
        bbb_permeant  = "Yes"  if (mw < 400 and logp > 0 and tpsa < 90) else "No"
        caco2         = round(0.5 - 0.01 * tpsa + 0.02 * logp, 3)  # rough estimate

        return {
            "ADMET_GI_Absorption":    gi_absorption,
            "ADMET_BBB_Permeant":     bbb_permeant,
            "ADMET_Caco2_approx":     caco2,
            "ADMET_MW":               round(mw, 3),
            "ADMET_LogP":             round(logp, 3),
            "ADMET_TPSA":             round(tpsa, 3),
            "ADMET_HBD":              hbd,
            "ADMET_HBA":              hba,
            "ADMET_RotBonds":         rot,
            "ADMET_NumRings":         rings,
            "ADMET_NumAromaticRings": arom,
            "ADMET_Lipinski_Pass":    int(mw<=500 and logp<=5 and hbd<=5 and hba<=10),
            "ADMET_Source":           "RDKit_fallback",
        }
    except Exception:
        return {}


def _fetch_admet_cached(args):
    """Worker for parallel pkCSM fetching. Returns (index, props)."""
    i, smiles = args
    cache_path = _cache_key("pkcms", smiles)
    cached = _load_cache(cache_path)
    if cached:
        return i, cached
    props = _with_retry(_fetch_pkcms_single, smiles) or {}
    _save_cache(cache_path, props)
    time.sleep(PKCMS_DELAY)  # pkCSM is a free server — be respectful
    return i, props


def enrich_with_admet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches ADMET properties in parallel using ThreadPoolExecutor.
    Uses 3 workers — pkCSM is a free academic server so we stay conservative.
    Same data, same results — just faster.
    """
    print("[*] pkCSM: fetching ADMET properties (parallel, 3 workers)...")
    smiles_list = list(df["SMILES"])
    results = [None] * len(smiles_list)
    completed = 0

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_fetch_admet_cached, (i, s)): i
                   for i, s in enumerate(smiles_list)}
        for future in as_completed(futures):
            i, props = future.result()
            results[i] = props
            completed += 1
            if completed % 10 == 0:
                print(f"    ... {completed}/{len(smiles_list)} done")

    admet_df = pd.DataFrame(results)
    print(f"    [✓] ADMET properties added ({len(admet_df.columns)} columns).")
    return pd.concat([df.reset_index(drop=True), admet_df.reset_index(drop=True)], axis=1)


# ==========================================
# UNICHEM — cross-reference SMILES to other DB IDs
# UniChem is a free EBI service mapping between
# chemical databases. We use it to find ChemSpider IDs
# and ZINC IDs from InChIKey for cross-referencing.
# ==========================================
def _smiles_to_inchikey(smiles: str) -> str:
    """Convert SMILES to InChIKey using RDKit."""
    try:
        from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        inchi = MolToInchi(mol)
        return InchiToInchiKey(inchi) if inchi else ""
    except:
        return ""


def _fetch_unichem_ids(inchikey: str) -> dict:
    """
    Fetch cross-database IDs from UniChem using multiple API strategies.
    UniChem has changed its API structure over time — we try v2, v1, and
    a PubChem-based fallback to ensure we always get cross-DB IDs.

    Strategy 1: UniChem v2 REST API (current)
    Strategy 2: UniChem v1 REST API (legacy fallback)
    Strategy 3: PubChem PUG REST — get CID, SID, synonyms from InChIKey
    """
    if not inchikey:
        return {}

    src_map = {
        "1":  "UC_ChEMBL_ID",
        "2":  "UC_DrugBank_ID",
        "12": "UC_ChemSpider_ID",
        "9":  "UC_ZINC_ID",
        "22": "UC_PubChem_SID",
        "7":  "UC_ChEBI_ID",
        "17": "UC_BindingDB_ID",
        "38": "UC_SureChEMBL_ID",
        "14": "UC_FDA_SRS_ID",
        "41": "UC_Comptox_ID",
    }

    # Strategy 1: UniChem v2
    try:
        url = f"https://www.ebi.ac.uk/unichem/api/v1/compounds?inchiKey={inchikey}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            sources = {}
            # v2 response: {"compounds": [{"sources": [...]}]}
            compounds = data.get("compounds", [data]) if "compounds" in data else [data]
            for compound in compounds:
                for src in compound.get("sources", []):
                    src_id = str(src.get("sourceId", src.get("src_id", "")))
                    cmp_id = src.get("compoundId", src.get("src_compound_id", ""))
                    if src_id in src_map and cmp_id:
                        sources[src_map[src_id]] = cmp_id
            if sources:
                return sources
    except Exception:
        pass

    # Strategy 2: UniChem v1 legacy
    try:
        url2 = f"https://www.ebi.ac.uk/unichem/rest/inchikey/{inchikey}"
        resp2 = requests.get(url2, timeout=10)
        if resp2.status_code == 200:
            data2 = resp2.json()
            sources2 = {}
            for item in data2:
                src_id = str(item.get("src_id", ""))
                cmp_id = item.get("src_compound_id", "")
                if src_id in src_map and cmp_id:
                    sources2[src_map[src_id]] = cmp_id
            if sources2:
                return sources2
    except Exception:
        pass

    # Strategy 3: PubChem fallback — get CID and synonyms from InChIKey
    try:
        pc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/IUPACName/JSON"
        pc_resp = requests.get(pc_url, timeout=10)
        if pc_resp.status_code == 200:
            pc_data = pc_resp.json()
            props = pc_data.get("PropertyTable", {}).get("Properties", [{}])[0]
            cid = props.get("CID", "")
            result = {}
            if cid:
                result["UC_PubChem_CID_from_IK"] = str(cid)
                # Also try to get SID from PubChem
                sid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/sids/JSON"
                sid_resp = requests.get(sid_url, timeout=10)
                if sid_resp.status_code == 200:
                    sids = sid_resp.json().get("InformationList", {}).get("Information", [{}])[0].get("SID", [])
                    result["UC_PubChem_SID"] = str(sids[0]) if sids else ""
            return result
    except Exception:
        pass

    return {}


def _fetch_unichem_cached(args):
    """Worker for parallel UniChem fetching. Returns (index, ids_dict)."""
    i, smiles = args
    cache_path = _cache_key("unichem", smiles)
    cached = _load_cache(cache_path)
    if cached:
        return i, cached
    inchikey = _smiles_to_inchikey(smiles)
    ids = _fetch_unichem_ids(inchikey)
    ids["InChIKey"] = inchikey
    _save_cache(cache_path, ids)
    time.sleep(0.1)
    return i, ids


def enrich_with_unichem(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cross-database IDs and InChIKey to every drug in parallel.
    Uses 8 workers — UniChem is a robust EBI API that handles concurrency well.
    """
    print("[*] UniChem: fetching cross-database IDs (parallel, 8 workers)...")
    smiles_list = list(df["SMILES"])
    results = [None] * len(smiles_list)
    completed = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_unichem_cached, (i, s)): i
                   for i, s in enumerate(smiles_list)}
        for future in as_completed(futures):
            i, ids = future.result()
            results[i] = ids
            completed += 1
            if completed % 10 == 0:
                print(f"    ... {completed}/{len(smiles_list)} done")

    unichem_df = pd.DataFrame(results)
    print(f"    [✓] UniChem IDs added ({len(unichem_df.columns)} columns).")
    return pd.concat([df.reset_index(drop=True), unichem_df.reset_index(drop=True)], axis=1)


# ==========================================
# CHEMBL BIOACTIVITY — IC50, Ki, EC50 per drug
# These are experimental activity values from ChEMBL
# assays — more informative than just structural props.
# ==========================================
def fetch_chembl_bioactivity(chembl_ids: list) -> pd.DataFrame:
    """
    For each drug, fetches best available bioactivity value
    (IC50, Ki, EC50, Kd) from ChEMBL assays.
    Returns DataFrame indexed by ChEMBL_ID with activity columns.
    """
    print("[*] ChEMBL: fetching bioactivity data (IC50/Ki/EC50)...")
    activity_api = new_client.activity
    records = {}

    for cid in chembl_ids:
        cache_path = _cache_key("bioact", cid)
        cached = _load_cache(cache_path)
        if cached:
            records[cid] = cached
            continue
        try:
            acts = activity_api.filter(
                molecule_chembl_id=cid,
                standard_type__in=["IC50", "Ki", "EC50", "Kd"],
            ).only(["standard_type", "standard_value", "standard_units", "assay_type"])
            best = {"BIO_IC50": None, "BIO_Ki": None, "BIO_EC50": None, "BIO_Kd": None}
            for a in acts:
                key = f"BIO_{a.get('standard_type','')}"
                val = a.get("standard_value")
                if key in best and best[key] is None and val:
                    try:
                        best[key] = float(val)
                    except:
                        pass
            _save_cache(cache_path, best)
            records[cid] = best
            time.sleep(0.1)
        except:
            records[cid] = {"BIO_IC50": None, "BIO_Ki": None, "BIO_EC50": None, "BIO_Kd": None}

    bio_df = pd.DataFrame.from_dict(records, orient="index").reset_index()
    bio_df.rename(columns={"index": "ChEMBL_ID"}, inplace=True)
    print(f"    [✓] Bioactivity data fetched for {len(bio_df)} drugs.")
    return bio_df


# ==========================================
# FULL FETCH PIPELINE
# ==========================================
def fetch_all(disease_name: str, max_drugs: int = MAX_DRUGS) -> pd.DataFrame:
    """
    Orchestrates all data sources in order:
    1. ChEMBL  — drug discovery + base RDKit properties
    2. PubChem — 20 physicochemical properties
    3. pkCSM   — ADMET pharmacokinetic properties
    4. UniChem — cross-database IDs (ChemSpider, ZINC, DrugBank, ChEBI)
    5. ChEMBL Bioactivity — IC50, Ki, EC50, Kd values
    Returns a single merged DataFrame ready for ML.
    """
    df = fetch_chembl_drugs(disease_name, max_drugs)
    if df.empty:
        return df
    df = enrich_with_pubchem(df)
    df = enrich_with_admet(df)
    df = enrich_with_unichem(df)

    # Bioactivity — merge on ChEMBL_ID
    if "ChEMBL_ID" in df.columns:
        bio_df = fetch_chembl_bioactivity(df["ChEMBL_ID"].tolist())
        df = df.merge(bio_df, on="ChEMBL_ID", how="left")

    return df
