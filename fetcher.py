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
# FULL FETCH PIPELINE
# ==========================================
def fetch_all(disease_name: str, max_drugs: int = MAX_DRUGS) -> pd.DataFrame:
    """
    Orchestrates all three data sources in order.
    Returns a single merged DataFrame ready for ML.
    """
    df = fetch_chembl_drugs(disease_name, max_drugs)
    if df.empty:
        return df
    df = enrich_with_pubchem(df)
    df = enrich_with_admet(df)
    return df
