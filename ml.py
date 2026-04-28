# ==========================================
# ml.py — topological indices, ML with k-fold CV,
#          SHAP feature importance, Lipinski filters
# ==========================================

import math
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

import shap
warnings.filterwarnings("ignore")

from config import TOPO_INDICES, ML_TARGETS, CV_FOLDS, RANDOM_STATE


# ==========================================
# ALGORITHM 1: Topological Indices
# ==========================================
def compute_topological_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 6 graph-theoretic topological indices per molecule:
      M1  — First Zagreb index (sum of squared vertex degrees)
      M2  — Second Zagreb index (sum of edge degree products)
      ABC — Atom-bond connectivity index
      R   — Randic connectivity index
      H   — Harmonic index
      F   — Forgotten topological index (sum of cubed degrees)
    """
    print("[*] Computing topological indices...")
    results = {k: [] for k in TOPO_INDICES}

    for smiles in df["SMILES"]:
        try:
            mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
            m1, m2, abc, r, h, f = 0, 0, 0, 0, 0, 0

            for atom in mol.GetAtoms():
                d = atom.GetDegree()
                m1 += d ** 2
                f  += d ** 3

            for bond in mol.GetBonds():
                u = bond.GetBeginAtom().GetDegree()
                v = bond.GetEndAtom().GetDegree()
                if u * v == 0:
                    continue
                m2  += u * v
                abc += math.sqrt((u + v - 2) / (u * v))
                r   += 1 / math.sqrt(u * v)
                h   += 2 / (u + v)

            for k, val in zip(TOPO_INDICES, [m1, m2, abc, r, h, f]):
                results[k].append(val)
        except:
            for k in TOPO_INDICES:
                results[k].append(np.nan)

    for k, vals in results.items():
        df[k] = vals

    df.dropna(subset=TOPO_INDICES, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"    [✓] Topological indices computed for {len(df)} molecules.")
    return df


# ==========================================
# ALGORITHM 2: Pearson Correlation
# ==========================================
def run_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation matrix between topological indices
    and physicochemical properties.
    """
    print("[*] Computing Pearson correlation...")
    available_targets = [t for t in ML_TARGETS if t in df.columns]
    corr = df[TOPO_INDICES + available_targets].corr().loc[TOPO_INDICES, available_targets]
    print("    [✓] Correlation matrix computed.")
    return corr


# ==========================================
# ALGORITHM 3: ML with k-fold CV + hyperparameter tuning
# ==========================================
MODELS_AND_GRIDS = {
    "LinearReg": (
        LinearRegression(),
        {}
    ),
    "RandomForest": (
        RandomForestRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20]}
    ),
    "XGBoost": (
        XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 6]}
    ),
    "NeuralNet": (
        MLPRegressor(max_iter=1000, random_state=RANDOM_STATE),
        {"model__hidden_layer_sizes": [(64, 32), (128, 64)], "model__alpha": [0.0001, 0.001]}
    ),
}


def run_ml_qspr(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Trains 4 ML models per property using k-fold CV and
    GridSearchCV for hyperparameter tuning.

    Returns:
      - results_df: DataFrame with R2, MAE, std for each model/property
      - best_models: dict of {property: fitted best model} for SHAP
    """
    print(f"[*] Running ML with {CV_FOLDS}-fold CV + hyperparameter tuning...")
    X = df[TOPO_INDICES].values
    available_targets = [t for t in ML_TARGETS if t in df.columns]
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_results = []
    best_models = {}

    for prop in available_targets:
        y = df[prop].values
        best_r2 = -np.inf
        best_model_for_prop = None

        for name, (model, param_grid) in MODELS_AND_GRIDS.items():
            # Pipeline: scale → model
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])

            if param_grid:
                gs = GridSearchCV(
                    pipe, param_grid,
                    cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
                    scoring="r2", n_jobs=-1
                )
                gs.fit(X, y)
                tuned_pipe = gs.best_estimator_
            else:
                tuned_pipe = pipe

            # k-fold CV on the (tuned) pipeline
            cv_r2  = cross_val_score(tuned_pipe, X, y, cv=kf, scoring="r2")
            cv_mae = cross_val_score(tuned_pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")

            mean_r2  = float(np.mean(cv_r2))
            std_r2   = float(np.std(cv_r2))
            mean_mae = float(-np.mean(cv_mae))

            all_results.append({
                "Property": prop,
                "Model":    name,
                "R2_mean":  round(mean_r2, 4),
                "R2_std":   round(std_r2, 4),
                "MAE_mean": round(mean_mae, 4),
            })

            if mean_r2 > best_r2:
                best_r2 = mean_r2
                tuned_pipe.fit(X, y)
                best_model_for_prop = tuned_pipe

        best_models[prop] = best_model_for_prop
        print(f"    [✓] {prop}: best R²={best_r2:.3f}")

    results_df = pd.DataFrame(all_results)
    print("[✓] ML complete.")
    return results_df, best_models


# ==========================================
# SHAP Feature Importance
# ==========================================
def compute_shap(df: pd.DataFrame, best_models: dict) -> dict:
    """
    Computes SHAP values for the best model of each property.
    Returns dict of {property: mean_abs_shap per feature}.
    Uses TreeExplainer for RF/XGBoost, KernelExplainer for others.
    """
    print("[*] Computing SHAP feature importance...")
    X = df[TOPO_INDICES].values
    shap_summaries = {}

    for prop, model in best_models.items():
        if model is None:
            continue
        try:
            inner = model.named_steps["model"]
            X_scaled = model.named_steps["scaler"].transform(X)

            if isinstance(inner, (RandomForestRegressor, XGBRegressor)):
                explainer = shap.TreeExplainer(inner)
                shap_vals = explainer.shap_values(X_scaled)
            else:
                bg = shap.kmeans(X_scaled, min(10, len(X_scaled)))
                explainer = shap.KernelExplainer(inner.predict, bg)
                shap_vals = explainer.shap_values(X_scaled, nsamples=50)

            mean_abs = np.abs(shap_vals).mean(axis=0)
            shap_summaries[prop] = {
                feat: round(float(val), 6)
                for feat, val in zip(TOPO_INDICES, mean_abs)
            }
            print(f"    [✓] SHAP done for {prop}")
        except Exception as e:
            print(f"    [!] SHAP failed for {prop}: {e}")

    return shap_summaries


# ==========================================
# LIPINSKI + DRUG-LIKENESS FILTERS
# ==========================================
def apply_drug_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies three drug-likeness filters using RDKit:

    Lipinski Rule of Five (oral bioavailability):
      MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10

    Veber rules (oral bioavailability, stricter):
      RotBonds ≤ 10, TPSA ≤ 140

    PAINS filter (pan-assay interference compounds):
      Flags problematic substructures that cause false positives
      in biological assays — important for publication credibility.
    """
    print("[*] Applying drug-likeness filters...")

    # Lipinski
    df["Lipinski_Pass"] = (
        (df["MolWt"]    <= 500) &
        (df["LogP"]     <= 5)   &
        (df["HBD"]      <= 5)   &
        (df["HBA"]      <= 10)
    )

    # Veber
    df["Veber_Pass"] = (
        (df["RotBonds"] <= 10) &
        (df["TPSA"]     <= 140)
    )

    # PAINS
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)

    pains_flags = []
    for smiles in df["SMILES"]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            pains_flags.append(not catalog.HasMatch(mol))  # True = passes (no PAINS)
        except:
            pains_flags.append(None)

    df["PAINS_Pass"] = pains_flags

    lip = df["Lipinski_Pass"].sum()
    veb = df["Veber_Pass"].sum()
    pai = pd.Series(pains_flags).eq(True).sum()
    total = len(df)

    print(f"    [✓] Lipinski: {lip}/{total} pass | Veber: {veb}/{total} pass | PAINS clean: {pai}/{total}")
    return df
