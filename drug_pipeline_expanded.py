# ==========================================
# INSTALL DEPENDENCIES
# ==========================================
# !pip install rdkit chembl_webresource_client xgboost scikit-learn pandas numpy requests

import math
import time
import requests
import numpy as np
import pandas as pd
import warnings
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from google.colab import files  # Uncomment if running in Colab

warnings.filterwarnings('ignore')


# ==========================================
# PRECURSOR: Extract Real Data (Disease X) — ChEMBL
# ==========================================
def extract_disease_data(disease_name, max_drugs=100):
    print(f"[*] Extracting real data for: '{disease_name}'...")
    indication_api = new_client.drug_indication
    indications = indication_api.filter(mesh_heading__icontains=disease_name).only(['molecule_chembl_id'])

    chembl_ids = list(set([ind['molecule_chembl_id'] for ind in indications]))
    if not chembl_ids:
        print(f"[!] No drugs found for '{disease_name}'.")
        return pd.DataFrame()

    molecule_api = new_client.molecule
    molecules = molecule_api.filter(molecule_chembl_id__in=chembl_ids[:max_drugs]).only(['molecule_chembl_id', 'molecule_structures'])

    data = []
    for mol in molecules:
        try:
            smiles = mol['molecule_structures']['canonical_smiles']
            rd_mol = Chem.MolFromSmiles(smiles)
            if rd_mol is None:
                continue
            data.append({
                'ChEMBL_ID': mol['molecule_chembl_id'],
                'SMILES': smiles,
                'MolWt': Descriptors.MolWt(rd_mol),
                'LogP': Descriptors.MolLogP(rd_mol),
                'TPSA': Descriptors.TPSA(rd_mol),
                'HBD': Descriptors.NumHDonors(rd_mol),
                'HBA': Descriptors.NumHAcceptors(rd_mol),
                'RotBonds': Descriptors.NumRotatableBonds(rd_mol),
                'MolMR': Descriptors.MolMR(rd_mol)
            })
        except:
            continue

    return pd.DataFrame(data)


# ==========================================
# NEW — SOURCE JUSTIFICATION NOTES
# ==========================================
# MolWt          → PubChem  (MolecularWeight, standardized computation)
# XLogP          → PubChem  (experimentally validated XLogP3)
# TPSA           → PubChem  (TPSA computed from canonical structure)
# HBD / HBA      → PubChem  (HBondDonorCount / HBondAcceptorCount, exact)
# RotBonds       → PubChem  (RotatableBondCount, standardized)
# Complexity     → PubChem  (only PubChem provides this)
# Charge         → PubChem  (FormalCharge, only PubChem provides this)
# HeavyAtomCount → PubChem  (only PubChem provides this)
# ExactMass      → PubChem  (MonoisotopicMass, high precision)
# MolFormula     → PubChem  (MolecularFormula)
# AtomStereo     → PubChem  (IsotopeAtomCount/StereoCount, unique to PubChem)
# CID            → PubChem  (identifier for cross-referencing)
# ADMET props    → pkCSM    (free REST, structure-based ADMET prediction)
# ==========================================


# ==========================================
# NEW — Fetch PubChem Physicochemical Properties
# ==========================================
PUBCHEM_PROPS = ",".join([
    "MolecularFormula",
    "MolecularWeight",
    "XLogP",
    "ExactMass",
    "MonoisotopicMass",
    "TPSA",
    "Complexity",
    "Charge",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
    "IsotopeAtomCount",
    "AtomStereoCount",
    "DefinedAtomStereoCount",
    "UndefinedAtomStereoCount",
    "BondStereoCount",
    "DefinedBondStereoCount",
    "UndefinedBondStereoCount",
    "CovalentUnitCount",
])

def fetch_pubchem_properties(smiles):
    """
    Given a SMILES string, fetch all available physicochemical
    properties from PubChem via their free REST API.
    Returns a dict of property name -> value, or empty dict on failure.

    Justification: PubChem is preferred over ChEMBL for physicochemical
    properties because it uses standardized structure representations,
    provides experimentally validated XLogP3, and offers unique properties
    like Complexity, FormalCharge, and stereo counts unavailable elsewhere.
    """
    try:
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
            f"property/{PUBCHEM_PROPS}/JSON"
        )
        resp = requests.post(url, data={"smiles": smiles}, timeout=15)
        if resp.status_code != 200:
            return {}
        props = resp.json()["PropertyTable"]["Properties"][0]

        return {
            "PC_CID":                     props.get("CID"),
            "PC_MolecularFormula":        props.get("MolecularFormula"),
            "PC_MolecularWeight":         props.get("MolecularWeight"),
            "PC_XLogP":                   props.get("XLogP"),
            "PC_ExactMass":               props.get("ExactMass"),
            "PC_MonoisotopicMass":        props.get("MonoisotopicMass"),
            "PC_TPSA":                    props.get("TPSA"),
            "PC_Complexity":              props.get("Complexity"),
            "PC_FormalCharge":            props.get("Charge"),
            "PC_HBD":                     props.get("HBondDonorCount"),
            "PC_HBA":                     props.get("HBondAcceptorCount"),
            "PC_RotatableBonds":          props.get("RotatableBondCount"),
            "PC_HeavyAtomCount":          props.get("HeavyAtomCount"),
            "PC_IsotopeAtomCount":        props.get("IsotopeAtomCount"),
            "PC_AtomStereoCount":         props.get("AtomStereoCount"),
            "PC_DefinedAtomStereoCount":  props.get("DefinedAtomStereoCount"),
            "PC_UndefinedAtomStereoCount":props.get("UndefinedAtomStereoCount"),
            "PC_BondStereoCount":         props.get("BondStereoCount"),
            "PC_DefinedBondStereoCount":  props.get("DefinedBondStereoCount"),
            "PC_UndefinedBondStereoCount":props.get("UndefinedBondStereoCount"),
            "PC_CovalentUnitCount":       props.get("CovalentUnitCount"),
        }
    except Exception as e:
        print(f"    [!] PubChem fetch error: {e}")
        return {}


def enrich_with_pubchem(df):
    """
    Iterates over all SMILES in df and adds PubChem properties as new columns.
    Rate-limited to ~5 requests/sec to stay within PubChem's free-tier limits.
    """
    print("[*] Fetching PubChem physicochemical properties...")
    all_props = []
    for i, smiles in enumerate(df['SMILES']):
        props = fetch_pubchem_properties(smiles)
        all_props.append(props)
        if (i + 1) % 10 == 0:
            print(f"    ... {i + 1}/{len(df)} done")
        time.sleep(0.2)  # 5 req/sec rate limit

    pubchem_df = pd.DataFrame(all_props)
    return pd.concat([df.reset_index(drop=True), pubchem_df.reset_index(drop=True)], axis=1)


# ==========================================
# NEW — Fetch ADMET Properties via pkCSM (free REST)
# ==========================================
# pkCSM provides ADMET predictions from SMILES for free.
# Properties include: absorption (Caco2, intestinal, P-gp substrate),
# distribution (VDss, BBB, CNS permeability), metabolism (CYP substrates/inhibitors),
# excretion (total clearance, renal OCT2), toxicity (AMES, hERG, LD50, etc.)
# ==========================================

def fetch_pkcms_admet(smiles):
    """
    Submits a SMILES to pkCSM and retrieves all predicted ADMET properties.
    pkCSM is a free, open REST API — no API key required.
    Returns a dict of ADMET property name -> value.
    """
    try:
        url = "https://biosig.lab.uq.edu.au/pkcsm/api/v1/prediction"
        payload = {"smiles": smiles}
        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            return {}
        result = resp.json()

        # Flatten all returned properties with ADMET_ prefix
        admet = {}
        for key, val in result.items():
            if key.lower() != "smiles":
                admet[f"ADMET_{key}"] = val
        return admet
    except Exception as e:
        print(f"    [!] pkCSM fetch error: {e}")
        return {}


def enrich_with_admet(df):
    """
    Iterates over all SMILES in df and adds pkCSM ADMET properties as new columns.
    """
    print("[*] Fetching ADMET properties via pkCSM...")
    all_admet = []
    for i, smiles in enumerate(df['SMILES']):
        props = fetch_pkcms_admet(smiles)
        all_admet.append(props)
        if (i + 1) % 10 == 0:
            print(f"    ... {i + 1}/{len(df)} done")
        time.sleep(0.5)  # be kind to the free server

    admet_df = pd.DataFrame(all_admet)
    return pd.concat([df.reset_index(drop=True), admet_df.reset_index(drop=True)], axis=1)


# ==========================================
# ALGORITHM 1: Compute Actual Topological Indices
# ==========================================
def compute_actual_indices(df):
    print("[*] Algorithm 1: Computing Actual Topological Indices...")
    results = {'M1': [], 'M2': [], 'ABC': [], 'R': [], 'H': [], 'F': []}

    for smiles in df['SMILES']:
        mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
        m1, m2, abc, r, h, f = 0, 0, 0, 0, 0, 0

        for atom in mol.GetAtoms():
            d = atom.GetDegree()
            m1 += d ** 2
            f += d ** 3

        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtom().GetDegree(), bond.GetEndAtom().GetDegree()
            if u * v == 0:
                continue
            m2 += (u * v)
            abc += math.sqrt((u + v - 2) / (u * v))
            r += 1 / math.sqrt(u * v)
            h += 2 / (u + v)

        for k, v_val in zip(results.keys(), [m1, m2, abc, r, h, f]):
            results[k].append(v_val)

    for k, v_list in results.items():
        df[k] = v_list
    return df


# ==========================================
# ALGORITHM 2: Pearson Correlation
# ==========================================
def run_correlation(df):
    print("[*] Algorithm 2: Running Pearson Correlation...")
    indices = ['M1', 'M2', 'ABC', 'R', 'H', 'F']
    props = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'MolMR']
    return df[indices + props].corr().loc[indices, props]


# ==========================================
# ALGORITHM 3: Predict Properties via ML
# ==========================================
def run_ml_qspr(df):
    print("[*] Algorithm 3: Training ML Models to predict Properties...")
    indices = ['M1', 'M2', 'ABC', 'R', 'H', 'F']
    props = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'MolMR']

    X = df[indices].values
    ml_results = []

    models = {
        'LinearReg': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'NeuralNet': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    }

    for p in props:
        y = df[p].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        for name, model in models.items():
            Xt, Xv = (X_train_s, X_test_s) if name in ['NeuralNet', 'XGBoost'] else (X_train, X_test)
            model.fit(Xt, y_train)
            pred = model.predict(Xv)
            ml_results.append({
                'Property': p, 'Model': name,
                'R2': r2_score(y_test, pred),
                'MAE': mean_absolute_error(y_test, pred)
            })

    return pd.DataFrame(ml_results)


# ==========================================
# RUN EVERYTHING
# ==========================================
DISEASE = "diabetes"  # Set any disease here

df_main = extract_disease_data(DISEASE)

if not df_main.empty:
    # Step 1: Add PubChem physicochemical properties
    df_main = enrich_with_pubchem(df_main)

    # Step 2: Add ADMET pharmacokinetic properties via pkCSM
    df_main = enrich_with_admet(df_main)

    # Step 3: Compute topological indices
    df_main = compute_actual_indices(df_main)

    # Step 4: Pearson correlation (on original ChEMBL-derived properties)
    corr_results = run_correlation(df_main)
    print("\n--- Pearson Correlation Matrix (Indices vs Properties) ---")
    print(corr_results.round(3))

    # Step 5: ML QSPR
    qspr_stats = run_ml_qspr(df_main)
    print("\n--- ML Prediction Performance (Best Model per Property) ---")
    top_models = qspr_stats.sort_values('R2', ascending=False).drop_duplicates('Property')
    print(top_models)

    # Export
    filename = f"{DISEASE}_final_results.csv"
    df_main.to_csv(filename, index=False)
    # files.download(filename)  # Uncomment in Colab
    print(f"\n[*] Pipeline Complete. Data exported to '{filename}'")
    print(f"[*] Total columns in output: {len(df_main.columns)}")
    print(f"[*] Columns: {list(df_main.columns)}")
