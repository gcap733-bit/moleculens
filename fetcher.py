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


def enrich_with_pubchem(df: pd.DataFrame) -> pd.DataFrame:
    print("[*] PubChem: fetching physicochemical properties...")
    all_props = []
    for i, smiles in enumerate(df["SMILES"]):
        cache_path = _cache_key("pubchem", smiles)
        cached = _load_cache(cache_path)
        if cached:
            all_props.append(cached)
        else:
            props = _with_retry(_fetch_pubchem_single, smiles) or {}
            _save_cache(cache_path, props)
            all_props.append(props)
            time.sleep(PUBCHEM_DELAY)
        if (i + 1) % 10 == 0:
            print(f"    ... {i+1}/{len(df)} done")

    pubchem_df = pd.DataFrame(all_props)
    print(f"    [✓] PubChem properties added ({len(pubchem_df.columns)} columns).")
    return pd.concat([df.reset_index(drop=True), pubchem_df.reset_index(drop=True)], axis=1)


# ==========================================
# pkCSM — ADMET pharmacokinetic properties
# Only free REST API with full ADMET predictions
# from SMILES. No API key required.
# ==========================================
def _fetch_pkcms_single(smiles: str) -> dict:
    url = "https://biosig.lab.uq.edu.au/pkcsm/api/v1/prediction"
    resp = requests.post(
        url,
        json={"smiles": smiles},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    return {
        f"ADMET_{k}": v
        for k, v in result.items()
        if k.lower() != "smiles"
    }


def enrich_with_admet(df: pd.DataFrame) -> pd.DataFrame:
    print("[*] pkCSM: fetching ADMET properties...")
    all_admet = []
    for i, smiles in enumerate(df["SMILES"]):
        cache_path = _cache_key("pkcms", smiles)
        cached = _load_cache(cache_path)
        if cached:
            all_admet.append(cached)
        else:
            props = _with_retry(_fetch_pkcms_single, smiles) or {}
            _save_cache(cache_path, props)
            all_admet.append(props)
            time.sleep(PKCMS_DELAY)
        if (i + 1) % 10 == 0:
            print(f"    ... {i+1}/{len(df)} done")

    admet_df = pd.DataFrame(all_admet)
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
    Fetch cross-database IDs from UniChem (free EBI API).
    Returns dict with ZINC ID, ChemSpider ID, DrugBank ID, etc.
    UniChem is preferred for ID mapping because it covers 40+
    chemical databases and is maintained by EMBL-EBI.
    """
    if not inchikey:
        return {}
    try:
        url = f"https://www.ebi.ac.uk/unichem/api/v1/compounds?inchiKey={inchikey}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        sources = {}
        src_map = {
            "1":  "UC_ChEMBL_ID",
            "2":  "UC_DrugBank_ID",
            "12": "UC_ChemSpider_ID",
            "9":  "UC_ZINC_ID",
            "22": "UC_PubChem_SID",
            "7":  "UC_ChEBI_ID",
            "17": "UC_BindingDB_ID",
            "38": "UC_SureChEMBL_ID",
        }
        for src in data.get("sources", []):
            src_id = str(src.get("sourceId", ""))
            if src_id in src_map:
                sources[src_map[src_id]] = src.get("compoundId", "")
        return sources
    except:
        return {}


def enrich_with_unichem(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cross-database IDs and InChIKey to every drug.
    Enables linking to ChemSpider, ZINC, DrugBank, ChEBI.
    """
    print("[*] UniChem: fetching cross-database IDs...")
    all_ids = []
    for i, smiles in enumerate(df["SMILES"]):
        cache_path = _cache_key("unichem", smiles)
        cached = _load_cache(cache_path)
        if cached:
            all_ids.append(cached)
        else:
            inchikey = _smiles_to_inchikey(smiles)
            ids = _fetch_unichem_ids(inchikey)
            ids["InChIKey"] = inchikey
            _save_cache(cache_path, ids)
            all_ids.append(ids)
            time.sleep(0.1)
        if (i + 1) % 10 == 0:
            print(f"    ... {i+1}/{len(df)} done")
    unichem_df = pd.DataFrame(all_ids)
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
