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

from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

import shap
warnings.filterwarnings("ignore")

from config import TOPO_INDICES, ML_TARGETS, CV_FOLDS, RANDOM_STATE


# ==========================================
# DISTANCE MATRIX HELPER
# ==========================================
def _distance_matrix(mol):
    n = mol.GetNumAtoms()
    INF = float('inf')
    dist = [[INF]*n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dist[u][v] = dist[v][u] = 1
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist


# ==========================================
# ALGORITHM 1: Topological Indices (14 total)
# ==========================================
def compute_topological_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 14 topological indices per molecule:
    Degree-based: M1, M2, ABC, R, H, F, AZI, GA, SC
    Distance-based: W (Wiener), J (Balaban), Z (Hosoya), Sz (Szeged), GE (Graph Entropy)
    """
    print("[*] Computing 14 topological indices...")
    all_keys = TOPO_INDICES
    results = {k: [] for k in all_keys}

    for smiles in df["SMILES"]:
        try:
            mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
            n = mol.GetNumAtoms()
            m_bonds = mol.GetNumBonds()
            INF = float('inf')

            # Degree-based
            m1 = m2 = abc = r = h = f = azi = ga = sc = 0
            degrees = [a.GetDegree() for a in mol.GetAtoms()]
            deg_sum = sum(degrees)

            for d in degrees:
                m1 += d**2
                f  += d**3

            for bond in mol.GetBonds():
                u = bond.GetBeginAtom().GetDegree()
                v = bond.GetEndAtom().GetDegree()
                if u*v == 0: continue
                m2  += u*v
                r   += 1/math.sqrt(u*v)
                h   += 2/(u+v)
                sc  += 1/math.sqrt(u+v)
                ga  += 2*math.sqrt(u*v)/(u+v)
                if (u+v-2) > 0:
                    abc += math.sqrt((u+v-2)/(u*v))
                if (u+v-2) != 0:
                    azi += (u*v/(u+v-2))**3

            # Graph entropy (Shannon, degree-based)
            ge = 0.0
            if deg_sum > 0:
                for d in degrees:
                    if d > 0:
                        p = d/deg_sum
                        ge -= p*math.log2(p)

            # Distance matrix
            dist = _distance_matrix(mol)

            # Wiener index
            w = sum(dist[i][j] for i in range(n) for j in range(i+1,n) if dist[i][j] != INF)

            # Balaban J index
            s = [sum(dist[i][j] for j in range(n) if dist[i][j] != INF) for i in range(n)]
            cyclo = m_bonds - n + 2
            j_bal = 0
            if cyclo > 0:
                j_sum = sum(1/math.sqrt(s[bond.GetBeginAtomIdx()]*s[bond.GetEndAtomIdx()])
                            for bond in mol.GetBonds()
                            if s[bond.GetBeginAtomIdx()]>0 and s[bond.GetEndAtomIdx()]>0)
                j_bal = (m_bonds/cyclo)*j_sum

            # Hosoya Z index (number of matchings via DP)
            edge_list = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
            p = [0]*(m_bonds+1); p[0] = 1
            def count_match(idx, matched):
                if idx == len(edge_list): return
                count_match(idx+1, matched)
                u_e, v_e = edge_list[idx]
                if u_e not in matched and v_e not in matched:
                    matched.add(u_e); matched.add(v_e)
                    sz2 = len(matched)//2
                    if sz2 < len(p): p[sz2] += 1
                    count_match(idx+1, matched)
                    matched.remove(u_e); matched.remove(v_e)
            if len(edge_list) <= 18:
                count_match(0, set())
            z_hosoya = sum(p)

            # Szeged index
            sz = 0
            for bond in mol.GetBonds():
                ui, vi = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                n_u = sum(1 for k2 in range(n) if dist[ui][k2]!=INF and dist[vi][k2]!=INF and dist[ui][k2]<dist[vi][k2])
                n_v = sum(1 for k2 in range(n) if dist[ui][k2]!=INF and dist[vi][k2]!=INF and dist[vi][k2]<dist[ui][k2])
                sz += n_u*n_v

            vals = [m1, m2, abc, r, h, f, azi, ga, sc, w, j_bal, z_hosoya, sz, ge]
            for k, val in zip(all_keys, vals):
                results[k].append(val)

        except:
            for k in all_keys:
                results[k].append(np.nan)

    for k, vals in results.items():
        df[k] = vals
    df.dropna(subset=all_keys, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"    [✓] 14 topological indices computed for {len(df)} molecules.")
    return df


# ==========================================
# ALGORITHM 2: Pearson + Spearman Correlation with p-values
#              + Multivariate (VIF)
# ==========================================
def run_correlation(df: pd.DataFrame) -> dict:
    """
    Extended correlation analysis:
      1. Pearson r + two-tailed p-value + significance flag
      2. Spearman rho + p-value (non-parametric, handles non-linearity)
      3. Variance Inflation Factor (VIF) for multicollinearity among indices
    Returns a dict with keys: 'pearson', 'spearman', 'pearson_p',
    'spearman_p', 'significance', 'vif'
    """
    print("[*] Computing extended correlation analysis...")
    available_targets = [t for t in ML_TARGETS if t in df.columns]
    data = df[TOPO_INDICES + available_targets].dropna()

    pearson_r  = pd.DataFrame(index=TOPO_INDICES, columns=available_targets, dtype=float)
    pearson_p  = pd.DataFrame(index=TOPO_INDICES, columns=available_targets, dtype=float)
    spearman_r = pd.DataFrame(index=TOPO_INDICES, columns=available_targets, dtype=float)
    spearman_p = pd.DataFrame(index=TOPO_INDICES, columns=available_targets, dtype=float)
    sig_flags  = pd.DataFrame(index=TOPO_INDICES, columns=available_targets, dtype=str)

    for idx in TOPO_INDICES:
        for prop in available_targets:
            x = data[idx].values
            y = data[prop].values
            pr, pp = pearsonr(x, y)
            sr, sp = spearmanr(x, y)
            pearson_r.loc[idx, prop]  = round(pr, 4)
            pearson_p.loc[idx, prop]  = round(pp, 4)
            spearman_r.loc[idx, prop] = round(float(sr), 4)
            spearman_p.loc[idx, prop] = round(float(sp), 4)
            # Significance: *** p<0.001, ** p<0.01, * p<0.05, ns
            if pp < 0.001:   sig_flags.loc[idx, prop] = "***"
            elif pp < 0.01:  sig_flags.loc[idx, prop] = "**"
            elif pp < 0.05:  sig_flags.loc[idx, prop] = "*"
            else:            sig_flags.loc[idx, prop] = "ns"

    # Variance Inflation Factor — multicollinearity among topo indices
    # VIF_i = 1 / (1 - R²_i) where R²_i from regressing index_i on all others
    vif_data = data[TOPO_INDICES].copy()
    vif_scores = {}
    for i, col in enumerate(TOPO_INDICES):
        others = [c for c in TOPO_INDICES if c != col]
        if not others:
            vif_scores[col] = 1.0
            continue
        X_oth = vif_data[others].values
        y_col = vif_data[col].values
        try:
            from sklearn.linear_model import LinearRegression as LR
            r2 = LR().fit(X_oth, y_col).score(X_oth, y_col)
            vif_scores[col] = round(1 / (1 - r2) if r2 < 1 else float('inf'), 3)
        except:
            vif_scores[col] = None

    print("    [✓] Pearson + Spearman + p-values + VIF computed.")
    return {
        "pearson":    pearson_r.round(4).to_dict(),
        "pearson_p":  pearson_p.round(4).to_dict(),
        "spearman":   spearman_r.round(4).to_dict(),
        "spearman_p": spearman_p.round(4).to_dict(),
        "significance": sig_flags.to_dict(),
        "vif":        vif_scores,
    }


# ==========================================
# ALGORITHM 3: ML with k-fold CV + hyperparameter tuning
# ==========================================
MODELS_AND_GRIDS = {
    # Linear family
    "LinearReg": (
        LinearRegression(), {}
    ),
    "Ridge": (
        Ridge(random_state=RANDOM_STATE),
        {"model__alpha": [0.1, 1.0, 10.0]}
    ),
    "Lasso": (
        Lasso(random_state=RANDOM_STATE, max_iter=2000),
        {"model__alpha": [0.01, 0.1, 1.0]}
    ),
    "ElasticNet": (
        ElasticNet(random_state=RANDOM_STATE, max_iter=2000),
        {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.3, 0.5, 0.7]}
    ),
    # Tree-based
    "RandomForest": (
        RandomForestRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20]}
    ),
    "ExtraTrees": (
        ExtraTreesRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]}
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 5]}
    ),
    "XGBoost": (
        XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 6]}
    ),
    # Kernel / probabilistic
    "SVR": (
        SVR(),
        {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["rbf", "linear"], "model__gamma": ["scale", "auto"]}
    ),
    "GaussianProcess": (
        GaussianProcessRegressor(
            kernel=Matern(nu=1.5) + WhiteKernel(),
            random_state=RANDOM_STATE, normalize_y=True
        ),
        {}  # GP kernel params handled internally
    ),
    # Neural
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
