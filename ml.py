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
# ALGORITHM 1: Topological Indices (34 total)
# ==========================================
def compute_topological_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 34 topological indices per molecule across 4 families:

    Original degree-based (9): M1, M2, ABC, R, H, F, AZI, GA, SC
    New degree-based from literature (11): BM, TM, GH, GBM, GTM, HG, BMG, BMH, TMG, TMH, SDD
    Reverse-degree variants (9): RM1, RM2, RABC, RR, RH, RF, RGA, RBM, RSDD
      (use n+1-d(v) as the reverse degree for each vertex)
    Degree-sum variants (5): DS1, DS2, DSR, DSH, DSGA
      (use d(u)+d(v) as edge weight)
    Distance-based (5): W (Wiener), J (Balaban), Z (Hosoya), Sz (Szeged), GE (Graph Entropy)
    """
    print("[*] Computing 34 topological indices...")
    all_keys = TOPO_INDICES
    results = {k: [] for k in all_keys}

    for smiles in df["SMILES"]:
        try:
            mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
            n = mol.GetNumAtoms()
            m_bonds = mol.GetNumBonds()
            INF = float("inf")

            degrees = [a.GetDegree() for a in mol.GetAtoms()]
            deg_sum = sum(degrees)
            max_deg = max(degrees) if degrees else 1

            # ── Original degree-based ────────────────────────
            m1 = m2 = abc = r = h = f = azi = ga = sc = 0
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

            # ── New degree-based indices from image ──────────
            bm = tm = gh = gbm = gtm = hg = bmg = bmh = tmg = tmh = sdd = 0
            for bond in mol.GetBonds():
                u = bond.GetBeginAtom().GetDegree()
                v = bond.GetEndAtom().GetDegree()
                if u*v == 0: continue
                suv  = u + v
                pruv = u * v
                suv2 = u**2 + v**2
                squv = math.sqrt(pruv)

                bm  += suv + pruv                         # Bi-Zagreb
                tm  += suv2 + pruv                        # Tri-Zagreb
                gh  += squv * suv / 2                     # Geometric-Harmonic
                if (suv + pruv) != 0:
                    gbm += squv / (suv + pruv)            # Geometric Bi-Zagreb
                if (suv2 + pruv) != 0:
                    gtm += squv / (suv2 + pruv)           # Geometric Tri-Zagreb
                if squv * suv != 0:
                    hg  += 2 / (squv * suv)               # Harmonic-Geometric
                if squv != 0:
                    bmg += (suv + pruv) / squv            # Bi Zagreb-Geometric
                    tmg += (suv2 + pruv) / squv           # Tri Zagreb-Geometric
                    sdd += suv2 / pruv                    # Symmetric Degree Division
                bmh += (suv + pruv) * suv / 2             # Bi Zagreb-Harmonic
                tmh += (suv2 + pruv) * suv / 2            # Tri Zagreb-Harmonic

            # ── Reverse-degree variants ──────────────────────
            # Reverse degree: rd(v) = n + 1 - d(v)
            # This transforms high-degree hubs into low-degree nodes
            # capturing complementary structural information
            rev_deg = [n + 1 - d for d in degrees]
            rm1 = rm2 = rabc = rr = rh = rf = rga = rbm = rsdd = 0
            for d in rev_deg:
                rm1 += d**2
                rf  += d**3
            for bond in mol.GetBonds():
                ru = rev_deg[bond.GetBeginAtomIdx()]
                rv = rev_deg[bond.GetEndAtomIdx()]
                if ru*rv == 0: continue
                rsuv  = ru + rv
                rpruv = ru * rv
                rsuv2 = ru**2 + rv**2
                rsquv = math.sqrt(rpruv)
                rm2  += rpruv
                rr   += 1/rsquv
                rh   += 2/rsuv
                if rsquv != 0:
                    rga  += 2*rsquv/rsuv
                    rsdd += rsuv2/rpruv
                rbm  += rsuv + rpruv
                if (rsuv+rpruv) != 0:
                    rabc += math.sqrt(abs(rsuv+rpruv-2)/rpruv) if rpruv > 0 else 0

            # ── Degree-sum variants ──────────────────────────
            # Use d(u)+d(v) as the primary weight for each edge
            # Captures pair-wise connectivity strength
            ds1 = ds2 = dsr = dsh = dsga = 0
            for bond in mol.GetBonds():
                u = bond.GetBeginAtom().GetDegree()
                v = bond.GetEndAtom().GetDegree()
                s = u + v
                if s == 0: continue
                ds1  += s**2
                ds2  += s*s
                dsr  += 1/math.sqrt(s)
                dsh  += 2/s
                if u*v > 0:
                    dsga += 2*math.sqrt(u*v)/s

            # ── Graph entropy ────────────────────────────────
            ge = 0.0
            if deg_sum > 0:
                for d in degrees:
                    if d > 0:
                        p = d/deg_sum
                        ge -= p*math.log2(p)

            # ── Distance-based ───────────────────────────────
            dist = _distance_matrix(mol)
            w = sum(dist[i][j] for i in range(n) for j in range(i+1,n)
                    if dist[i][j] != INF)
            s_dist = [sum(dist[i][j] for j in range(n) if dist[i][j] != INF)
                      for i in range(n)]
            cyclo = m_bonds - n + 2
            j_bal = 0
            if cyclo > 0:
                j_sum = sum(1/math.sqrt(s_dist[bond.GetBeginAtomIdx()]*s_dist[bond.GetEndAtomIdx()])
                            for bond in mol.GetBonds()
                            if s_dist[bond.GetBeginAtomIdx()]>0 and s_dist[bond.GetEndAtomIdx()]>0)
                j_bal = (m_bonds/cyclo)*j_sum

            edge_list = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
            p_match = [0]*(m_bonds+1); p_match[0] = 1
            def count_match(idx, matched):
                if idx == len(edge_list): return
                count_match(idx+1, matched)
                u_e, v_e = edge_list[idx]
                if u_e not in matched and v_e not in matched:
                    matched.add(u_e); matched.add(v_e)
                    sz2 = len(matched)//2
                    if sz2 < len(p_match): p_match[sz2] += 1
                    count_match(idx+1, matched)
                    matched.remove(u_e); matched.remove(v_e)
            if len(edge_list) <= 18:
                count_match(0, set())
            z_hosoya = sum(p_match)

            sz = 0
            for bond in mol.GetBonds():
                ui, vi = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                n_u = sum(1 for k2 in range(n) if dist[ui][k2]!=INF and dist[vi][k2]!=INF and dist[ui][k2]<dist[vi][k2])
                n_v = sum(1 for k2 in range(n) if dist[ui][k2]!=INF and dist[vi][k2]!=INF and dist[vi][k2]<dist[ui][k2])
                sz += n_u*n_v

            vals = [
                # Original (9)
                m1, m2, abc, r, h, f, azi, ga, sc,
                # New from image (11)
                bm, tm, gh, gbm, gtm, hg, bmg, bmh, tmg, tmh, sdd,
                # Reverse-degree (9)
                rm1, rm2, rabc, rr, rh, rf, rga, rbm, rsdd,
                # Degree-sum (5)
                ds1, ds2, dsr, dsh, dsga,
                # Distance-based (5)
                w, j_bal, z_hosoya, sz, ge,
            ]

            for k, val in zip(all_keys, vals):
                results[k].append(val)

        except Exception as e:
            for k in all_keys:
                results[k].append(np.nan)

    for k, vals in results.items():
        # Replace inf/-inf with nan so dropna removes them cleanly
        clean = []
        for v in vals:
            try:
                if v != v or abs(v) == float('inf'):  # nan or inf check
                    clean.append(float('nan'))
                else:
                    clean.append(v)
            except:
                clean.append(float('nan'))
        df[k] = clean
    df.dropna(subset=all_keys, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"    [✓] 34 topological indices computed for {len(df)} molecules.")
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
    # Only use targets that exist AND have enough non-null values (>50% filled)
    available_targets = [
        t for t in ML_TARGETS
        if t in df.columns and df[t].notna().sum() >= max(10, len(df) * 0.5)
    ]
    print(f"    [*] Predicting {len(available_targets)} properties: {available_targets}")
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
