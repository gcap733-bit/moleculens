# ==========================================
# fetcher.py — data fetching with retries + disk caching
# Sources: ChEMBL, PubChem, pkCSM
# FIXED: ADMET timeout was catastrophic (360s/drug). Now uses RDKit fallback immediately.
# FIXED: ChEMBL indication fetching now stops early once max_drugs IDs collected.
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
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache(path: str, data):
    if CACHE_ENABLED:
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass


# ==========================================
# RETRY WRAPPER
# ==========================================
def _with_retry(fn, *args, **kwargs):
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
# FIXED: Stops collecting IDs early once we have enough.
# ==========================================
def fetch_chembl_drugs(disease_name: str, max_drugs: int = MAX_DRUGS) -> pd.DataFrame:
    print(f"[*] ChEMBL: fetching drugs for '{disease_name}'...")

    cache_path = _cache_key("chembl", disease_name)
    cached = _load_cache(cache_path)
    if cached:
        print(f"    [cache] Loaded {len(cached)} drugs from disk.")
        return pd.DataFrame(cached)

    chembl_ids = set()

    # Strategy 1: mesh_heading icontains — iterate and stop early
    try:
        indication_api = new_client.drug_indication
        qs = indication_api.filter(mesh_heading__icontains=disease_name).only(["molecule_chembl_id"])
        for ind in qs:
            chembl_ids.add(ind['molecule_chembl_id'])
            if len(chembl_ids) >= max_drugs * 2:
                break
        print(f"    [*] Strategy 1 (mesh): Found {len(chembl_ids)} unique ChEMBL IDs")
    except Exception as e:
        print(f"    [!] Strategy 1 failed: {e}")

    # Strategy 2: broad search if not enough
    if len(chembl_ids) < 20:
        print(f"    [*] Only {len(chembl_ids)} drugs found. Trying broader strategy...")
        try:
            indication_api = new_client.drug_indication
            qs2 = indication_api.search(disease_name).only(["molecule_chembl_id"])
            for ind in qs2:
                chembl_ids.add(ind['molecule_chembl_id'])
                if len(chembl_ids) >= max_drugs * 2:
                    break
            print(f"    [*] Strategy 2 (search): Now have {len(chembl_ids)} unique ChEMBL IDs")
        except Exception as e:
            print(f"    [!] Strategy 2 failed: {e}")

    # Strategy 3: individual disease keywords if still not enough
    if len(chembl_ids) < 30 and ' ' in disease_name:
        print(f"    [*] Trying individual keywords from '{disease_name}'...")
        keywords = disease_name.split()[:3]
        for keyword in keywords:
            if len(chembl_ids) >= max_drugs:
                break
            try:
                qs3 = new_client.drug_indication.search(keyword).only(["molecule_chembl_id"])
                for ind in qs3:
                    chembl_ids.add(ind['molecule_chembl_id'])
                    if len(chembl_ids) >= max_drugs:
                        break
                print(f"    [*] Keyword '{keyword}': Now have {len(chembl_ids)} IDs")
            except Exception as e:
                print(f"    [!] Keyword '{keyword}' failed: {e}")

    if not chembl_ids:
        print(f"    [!] No drugs found for '{disease_name}' after all strategies.")
        return pd.DataFrame()

    chembl_ids = list(chembl_ids)[:max_drugs]
    print(f"    [*] Fetching molecule structures for {len(chembl_ids)} IDs...")

    # Fetch molecule structures in batches
    BATCH = 50
    all_molecules = []

    def _fetch_batch(batch_ids):
        molecule_api = new_client.molecule
        return list(
            molecule_api.filter(
                molecule_chembl_id__in=batch_ids
            ).only(["molecule_chembl_id", "molecule_structures"])
        )

    for i in range(0, len(chembl_ids), BATCH):
        batch = chembl_ids[i:i + BATCH]
        mols = _with_retry(_fetch_batch, batch)
        if mols:
            all_molecules.extend(mols)
        time.sleep(0.1)

    if not all_molecules:
        print(f"    [!] No molecule structures retrieved.")
        return pd.DataFrame()

    # Compute RDKit descriptors
    data = []
    for mol in all_molecules:
        try:
            structs = mol.get('molecule_structures') or {}
            smiles = structs.get('canonical_smiles', '')
            if not smiles:
                continue
            rd_mol = Chem.MolFromSmiles(smiles)
            if rd_mol is None:
                continue
            data.append({
                "ChEMBL_ID": mol["molecule_chembl_id"],
                "SMILES":    smiles,
                "MolWt":     Descriptors.MolWt(rd_mol),
                "LogP":      Descriptors.MolLogP(rd_mol),
                "TPSA":      Descriptors.TPSA(rd_mol),
                "HBD":       Descriptors.NumHDonors(rd_mol),
                "HBA":       Descriptors.NumHAcceptors(rd_mol),
                "RotBonds":  Descriptors.NumRotatableBonds(rd_mol),
                "MolMR":     Descriptors.MolMR(rd_mol),
            })
        except Exception:
            continue

    _save_cache(cache_path, data)
    print(f"    [✓] {len(data)} drugs fetched and cached.")
    return pd.DataFrame(data)


# ==========================================
# PUBCHEM — 20 physicochemical properties
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
    i, smiles = args
    cache_path = _cache_key("pubchem", smiles)
    cached = _load_cache(cache_path)
    if cached:
        return i, cached
    props = _with_retry(_fetch_pubchem_single, smiles) or {}
    _save_cache(cache_path, props)
    time.sleep(PUBCHEM_DELAY)
    return i, props


def enrich_with_pubchem(df: pd.DataFrame) -> pd.DataFrame:
    print("[*] PubChem: fetching physicochemical properties (parallel, 5 workers)...")
    smiles_list = list(df["SMILES"])
    results = [None] * len(smiles_list)
    completed = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_pubchem_cached, (i, s)): i
                   for i, s in enumerate(smiles_list)}
        for future in as_completed(futures):
            try:
                i, props = future.result()
                results[i] = props
            except Exception:
                results[futures[future]] = {}
            completed += 1
            if completed % 10 == 0:
                print(f"    ... {completed}/{len(smiles_list)} done")

    results = [r if r is not None else {} for r in results]
    pubchem_df = pd.DataFrame(results)
    print(f"    [✓] PubChem properties added ({len(pubchem_df.columns)} columns).")
    return pd.concat([df.reset_index(drop=True), pubchem_df.reset_index(drop=True)], axis=1)


# ==========================================
# ADMET — RDKit-only (fast, no external API dependency)
# FIXED: Removed unreliable pkCSM external API calls that caused 30+ min timeouts.
# The biosig API is frequently down/slow. RDKit gives us reliable ADMET approximations.
# ==========================================
def _compute_admet_rdkit(smiles: str) -> dict:
    """Compute ADMET properties using RDKit descriptors only. Fast and reliable."""
    try:
        from rdkit.Chem import rdMolDescriptors
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
        qed   = Descriptors.qed(mol) if hasattr(Descriptors, 'qed') else None

        gi_absorption = "High" if (mw < 500 and logp < 5 and hbd <= 5 and hba <= 10) else "Low"
        bbb_permeant  = "Yes"  if (mw < 400 and logp > 0 and tpsa < 90) else "No"
        caco2         = round(0.5 - 0.01 * tpsa + 0.02 * logp, 3)

        result = {
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
            "ADMET_Source":           "RDKit",
        }
        if qed is not None:
            result["ADMET_QED"] = round(qed, 4)
        return result
    except Exception:
        return {}


def _fetch_admet_cached(args):
    i, smiles = args
    cache_path = _cache_key("admet_rdkit", smiles)
    cached = _load_cache(cache_path)
    if cached:
        return i, cached
    props = _compute_admet_rdkit(smiles)
    _save_cache(cache_path, props)
    return i, props


def enrich_with_admet(df: pd.DataFrame) -> pd.DataFrame:
    print("[*] Computing ADMET properties (RDKit, parallel, 8 workers)...")
    smiles_list = list(df["SMILES"])
    results = [None] * len(smiles_list)
    completed = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_admet_cached, (i, s)): i
                   for i, s in enumerate(smiles_list)}
        for future in as_completed(futures):
            try:
                i, props = future.result()
                results[i] = props
            except Exception:
                results[futures[future]] = {}
            completed += 1
            if completed % 20 == 0:
                print(f"    ... {completed}/{len(smiles_list)} done")

    results = [r if r is not None else {} for r in results]
    admet_df = pd.DataFrame(results)
    print(f"    [✓] ADMET properties added ({len(admet_df.columns)} columns).")
    return pd.concat([df.reset_index(drop=True), admet_df.reset_index(drop=True)], axis=1)


# ==========================================
# UNICHEM — cross-reference IDs
# ==========================================
def _smiles_to_inchikey(smiles: str) -> str:
    try:
        from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        inchi = MolToInchi(mol)
        return InchiToInchiKey(inchi) if inchi else ""
    except Exception:
        return ""


def _fetch_unichem_ids(inchikey: str) -> dict:
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
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            sources = {}
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
        resp2 = requests.get(url2, timeout=8)
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

    return {}


def _fetch_unichem_cached(args):
    i, smiles = args
    cache_path = _cache_key("unichem", smiles)
    cached = _load_cache(cache_path)
    if cached:
        return i, cached
    inchikey = _smiles_to_inchikey(smiles)
    ids = _fetch_unichem_ids(inchikey)
    ids["InChIKey"] = inchikey
    _save_cache(cache_path, ids)
    time.sleep(0.05)
    return i, ids


def enrich_with_unichem(df: pd.DataFrame) -> pd.DataFrame:
    print("[*] UniChem: fetching cross-database IDs (parallel, 8 workers)...")
    smiles_list = list(df["SMILES"])
    results = [None] * len(smiles_list)
    completed = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_unichem_cached, (i, s)): i
                   for i, s in enumerate(smiles_list)}
        for future in as_completed(futures):
            try:
                i, ids = future.result()
                results[i] = ids
            except Exception:
                results[futures[future]] = {}
            completed += 1
            if completed % 20 == 0:
                print(f"    ... {completed}/{len(smiles_list)} done")

    results = [r if r is not None else {} for r in results]
    unichem_df = pd.DataFrame(results)
    print(f"    [✓] UniChem IDs added ({len(unichem_df.columns)} columns).")
    return pd.concat([df.reset_index(drop=True), unichem_df.reset_index(drop=True)], axis=1)


# ==========================================
# CHEMBL BIOACTIVITY — IC50, Ki, EC50 per drug (parallelized)
# ==========================================
def _fetch_bioact_single(args):
    cid, cache_path = args
    cached = _load_cache(cache_path)
    if cached:
        return cid, cached
    try:
        activity_api = new_client.activity
        acts = list(activity_api.filter(
            molecule_chembl_id=cid,
            standard_type__in=["IC50", "Ki", "EC50", "Kd"],
        ).only(["standard_type", "standard_value", "standard_units", "assay_type"]))
        best = {"BIO_IC50": None, "BIO_Ki": None, "BIO_EC50": None, "BIO_Kd": None}
        for a in acts:
            key = f"BIO_{a.get('standard_type','')}"
            val = a.get("standard_value")
            if key in best and best[key] is None and val:
                try:
                    best[key] = float(val)
                except Exception:
                    pass
        _save_cache(cache_path, best)
        return cid, best
    except Exception:
        return cid, {"BIO_IC50": None, "BIO_Ki": None, "BIO_EC50": None, "BIO_Kd": None}


def fetch_chembl_bioactivity(chembl_ids: list) -> pd.DataFrame:
    print("[*] ChEMBL: fetching bioactivity data (IC50/Ki/EC50, parallel 4 workers)...")
    args_list = [(cid, _cache_key("bioact", cid)) for cid in chembl_ids]
    records = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_fetch_bioact_single, args): args[0] for args in args_list}
        completed = 0
        for future in as_completed(futures):
            try:
                cid, best = future.result()
                records[cid] = best
            except Exception as e:
                cid = futures[future]
                records[cid] = {"BIO_IC50": None, "BIO_Ki": None, "BIO_EC50": None, "BIO_Kd": None}
            completed += 1
            if completed % 20 == 0:
                print(f"    ... {completed}/{len(chembl_ids)} done")

    bio_df = pd.DataFrame.from_dict(records, orient="index").reset_index()
    bio_df.rename(columns={"index": "ChEMBL_ID"}, inplace=True)
    print(f"    [✓] Bioactivity data fetched for {len(bio_df)} drugs.")
    return bio_df


# ==========================================
# FULL FETCH PIPELINE
# ==========================================
def fetch_all(disease_name: str, max_drugs: int = MAX_DRUGS) -> pd.DataFrame:
    df = fetch_chembl_drugs(disease_name, max_drugs)
    if df.empty:
        return df
    df = enrich_with_pubchem(df)
    df = enrich_with_admet(df)
    df = enrich_with_unichem(df)

    if "ChEMBL_ID" in df.columns:
        bio_df = fetch_chembl_bioactivity(df["ChEMBL_ID"].tolist())
        df = df.merge(bio_df, on="ChEMBL_ID", how="left")

    # Coerce numeric columns
    skip_cols = {
        "ChEMBL_ID", "SMILES", "InChIKey", "PC_MolecularFormula",
        "UC_ChEMBL_ID", "UC_DrugBank_ID", "UC_ChemSpider_ID",
        "UC_ZINC_ID", "UC_PubChem_SID", "UC_ChEBI_ID",
        "UC_BindingDB_ID", "UC_SureChEMBL_ID", "UC_FDA_SRS_ID",
        "UC_Comptox_ID", "UC_PubChem_CID_from_IK", "ADMET_Source",
        "ADMET_GI_Absorption", "ADMET_BBB_Permeant"
    }
    for col in df.columns:
        if col not in skip_cols:
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().sum() > 0:
                    df[col] = converted
            except Exception:
                pass

    print(f"    [✓] Fetch complete: {len(df)} drugs, {len(df.columns)} columns.")
    return df

