# ==========================================
# ml.py — topological indices, ML, SHAP, filters
# Fully numpy-2.x compatible, crash-proof
# ==========================================

import math
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

import shap
warnings.filterwarnings("ignore")

from config import TOPO_INDICES, ML_TARGETS, CV_FOLDS, RANDOM_STATE


# ==========================================
# SAFE NUMERIC CONVERSION HELPERS
# ==========================================
def _safe_float(v):
    """Convert any value to float, returning NaN on failure."""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return float('nan')
        return f
    except Exception:
        return float('nan')


def _df_to_float_array(df, cols):
    """
    Convert selected columns of a DataFrame to a clean float64 numpy array.
    Handles object dtypes, inf, nan — fully numpy-2.x compatible.
    """
    arr = np.zeros((len(df), len(cols)), dtype=np.float64)
    for j, col in enumerate(cols):
        series = df[col]
        for i, val in enumerate(series):
            arr[i, j] = _safe_float(val)
    return arr


def _sanitise_df(df, cols):
    """
    Return a copy of df with only the given cols, all converted to float64.
    Drops any row that has NaN in any of the cols.
    """
    out = pd.DataFrame(index=df.index)
    for col in cols:
        out[col] = [_safe_float(v) for v in df[col]]
    out = out.dropna()
    return out


# ==========================================
# DISTANCE MATRIX HELPER
# ==========================================
def _distance_matrix(mol):
    n = mol.GetNumAtoms()
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dist[u][v] = dist[v][u] = 1
    for k in range(n):
        for i in range(n):
            for j in range(n):
                nd = dist[i][k] + dist[k][j]
                if nd < dist[i][j]:
                    dist[i][j] = nd
    return dist


# ==========================================
# ALGORITHM 1: 39 Topological Indices
# ==========================================
def compute_topological_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 39 topological indices per molecule across 5 families:
      Original degree-based (9): M1,M2,ABC,R,H,F,AZI,GA,SC
      New degree-based (11):     BM,TM,GH,GBM,GTM,HG,BMG,BMH,TMG,TMH,SDD
      Reverse-degree (9):        RM1,RM2,RABC,RR,RH,RF,RGA,RBM,RSDD
      Degree-sum (5):            DS1,DS2,DSR,DSH,DSGA
      Distance-based (5):        W,J,Z,Sz,GE
    """
    print("[*] Computing 39 topological indices...")
    all_keys = TOPO_INDICES
    results = {k: [] for k in all_keys}

    for smiles in df["SMILES"]:
        try:
            mol = Chem.RemoveHs(Chem.MolFromSmiles(str(smiles)))
            if mol is None:
                raise ValueError("Invalid SMILES")
            n = mol.GetNumAtoms()
            m_bonds = mol.GetNumBonds()
            INF = float("inf")

            degrees = [a.GetDegree() for a in mol.GetAtoms()]
            deg_sum = sum(degrees)

            # ── Original degree-based ──────────────────────
            m1 = m2 = abc = r = h = f = azi = ga = sc = 0.0
            for d in degrees:
                m1 += d * d
                f  += d * d * d

            for bond in mol.GetBonds():
                u = bond.GetBeginAtom().GetDegree()
                v = bond.GetEndAtom().GetDegree()
                if u == 0 or v == 0:
                    continue
                uv = u * v
                uv_s = u + v
                m2  += uv
                r   += 1.0 / math.sqrt(uv)
                h   += 2.0 / uv_s
                sc  += 1.0 / math.sqrt(uv_s)
                ga  += 2.0 * math.sqrt(uv) / uv_s
                if uv_s - 2 > 0:
                    abc += math.sqrt((uv_s - 2) / uv)
                if uv_s - 2 != 0:
                    azi += (uv / (uv_s - 2)) ** 3

            # ── New degree-based ───────────────────────────
            bm = tm = gh = gbm = gtm = hg = bmg = bmh = tmg = tmh = sdd = 0.0
            for bond in mol.GetBonds():
                u = bond.GetBeginAtom().GetDegree()
                v = bond.GetEndAtom().GetDegree()
                if u == 0 or v == 0:
                    continue
                suv   = float(u + v)
                pruv  = float(u * v)
                suv2  = float(u*u + v*v)
                squv  = math.sqrt(pruv)

                bm  += suv + pruv
                tm  += suv2 + pruv
                gh  += squv * suv / 2.0
                denom_gbm = suv + pruv
                if denom_gbm != 0:
                    gbm += squv / denom_gbm
                denom_gtm = suv2 + pruv
                if denom_gtm != 0:
                    gtm += squv / denom_gtm
                denom_hg = squv * suv
                if denom_hg != 0:
                    hg  += 2.0 / denom_hg
                if squv != 0:
                    bmg += (suv + pruv) / squv
                    tmg += (suv2 + pruv) / squv
                    sdd += suv2 / pruv
                bmh += (suv + pruv) * suv / 2.0
                tmh += (suv2 + pruv) * suv / 2.0

            # ── Reverse-degree ─────────────────────────────
            rev_deg = [n + 1 - d for d in degrees]
            rm1 = rm2 = rabc = rr = rh = rf = rga = rbm = rsdd = 0.0
            for d in rev_deg:
                rm1 += float(d * d)
                rf  += float(d * d * d)
            for bond in mol.GetBonds():
                ru = rev_deg[bond.GetBeginAtomIdx()]
                rv = rev_deg[bond.GetEndAtomIdx()]
                if ru == 0 or rv == 0:
                    continue
                rpruv = float(ru * rv)
                rsuv  = float(ru + rv)
                rsuv2 = float(ru*ru + rv*rv)
                rsquv = math.sqrt(rpruv)
                rm2  += rpruv
                rr   += 1.0 / rsquv
                rh   += 2.0 / rsuv
                if rsquv != 0:
                    rga  += 2.0 * rsquv / rsuv
                    rsdd += rsuv2 / rpruv
                rbm  += rsuv + rpruv
                num_rabc = rsuv + rpruv - 2.0
                if rpruv > 0 and num_rabc > 0:
                    rabc += math.sqrt(num_rabc / rpruv)

            # ── Degree-sum ─────────────────────────────────
            ds1 = ds2 = dsr = dsh = dsga = 0.0
            for bond in mol.GetBonds():
                u = bond.GetBeginAtom().GetDegree()
                v = bond.GetEndAtom().GetDegree()
                s = float(u + v)
                if s == 0:
                    continue
                ds1  += s * s
                ds2  += s * s
                dsr  += 1.0 / math.sqrt(s)
                dsh  += 2.0 / s
                uv = u * v
                if uv > 0:
                    dsga += 2.0 * math.sqrt(uv) / s

            # ── Graph entropy ──────────────────────────────
            ge = 0.0
            if deg_sum > 0:
                for d in degrees:
                    if d > 0:
                        p = d / float(deg_sum)
                        ge -= p * math.log2(p)

            # ── Distance-based ─────────────────────────────
            dist = _distance_matrix(mol)
            w = sum(
                dist[i][j]
                for i in range(n)
                for j in range(i + 1, n)
                if dist[i][j] != INF
            )
            s_dist = [
                sum(dist[i][j] for j in range(n) if dist[i][j] != INF)
                for i in range(n)
            ]
            cyclo = m_bonds - n + 2
            j_bal = 0.0
            if cyclo > 0:
                j_sum = 0.0
                for bond in mol.GetBonds():
                    bi = bond.GetBeginAtomIdx()
                    ei = bond.GetEndAtomIdx()
                    sb = s_dist[bi]
                    se = s_dist[ei]
                    if sb > 0 and se > 0:
                        j_sum += 1.0 / math.sqrt(sb * se)
                j_bal = (m_bonds / cyclo) * j_sum

            # Hosoya Z
            edge_list = [
                (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()
            ]
            p_match = [0] * (m_bonds + 1)
            p_match[0] = 1

            def count_match(idx, matched):
                if idx == len(edge_list):
                    return
                count_match(idx + 1, matched)
                ue, ve = edge_list[idx]
                if ue not in matched and ve not in matched:
                    matched.add(ue)
                    matched.add(ve)
                    sz2 = len(matched) // 2
                    if sz2 < len(p_match):
                        p_match[sz2] += 1
                    count_match(idx + 1, matched)
                    matched.remove(ue)
                    matched.remove(ve)

            if len(edge_list) <= 18:
                count_match(0, set())
            z_hosoya = float(sum(p_match))

            # Szeged
            sz = 0.0
            for bond in mol.GetBonds():
                ui = bond.GetBeginAtomIdx()
                vi = bond.GetEndAtomIdx()
                n_u = sum(
                    1 for k2 in range(n)
                    if dist[ui][k2] != INF
                    and dist[vi][k2] != INF
                    and dist[ui][k2] < dist[vi][k2]
                )
                n_v = sum(
                    1 for k2 in range(n)
                    if dist[ui][k2] != INF
                    and dist[vi][k2] != INF
                    and dist[vi][k2] < dist[ui][k2]
                )
                sz += float(n_u * n_v)

            vals = [
                m1, m2, abc, r, h, f, azi, ga, sc,        # 9 original
                bm, tm, gh, gbm, gtm, hg, bmg, bmh, tmg, tmh, sdd,  # 11 new
                rm1, rm2, rabc, rr, rh, rf, rga, rbm, rsdd,  # 9 reverse
                ds1, ds2, dsr, dsh, dsga,                  # 5 degree-sum
                w, j_bal, z_hosoya, sz, ge,                # 5 distance
            ]

            for k, val in zip(all_keys, vals):
                results[k].append(_safe_float(val))

        except Exception:
            for k in all_keys:
                results[k].append(float('nan'))

    for k, vals in results.items():
        df[k] = [v if not math.isnan(v) else float('nan') for v in vals]

    df.dropna(subset=all_keys, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"    [\u2713] 39 topological indices computed for {len(df)} molecules.")
    return df


# ==========================================
# ALGORITHM 2: Correlation (Pearson + Spearman + VIF)
# ==========================================
def run_correlation(df: pd.DataFrame) -> dict:
    print("[*] Computing correlation analysis...")

    valid_topo = [
        t for t in TOPO_INDICES
        if t in df.columns
    ]
    available_targets = [
        t for t in ML_TARGETS
        if t in df.columns
        and df[t].apply(lambda x: _safe_float(x)).notna().sum() >= max(5, len(df) * 0.3)
    ]

    if not available_targets or not valid_topo:
        print("    [!] No valid targets or indices for correlation.")
        return {"pearson": {}, "pearson_p": {}, "spearman": {}, "spearman_p": {},
                "significance": {}, "vif": {}}

    # Build clean numeric arrays
    clean_data = {}
    for col in valid_topo + available_targets:
        clean_data[col] = [_safe_float(v) for v in df[col]]

    clean_df = pd.DataFrame(clean_data).dropna()
    if len(clean_df) < 5:
        print("    [!] Insufficient data for correlation.")
        return {"pearson": {}, "pearson_p": {}, "spearman": {}, "spearman_p": {},
                "significance": {}, "vif": {}}

    pearson_r  = {}
    pearson_p  = {}
    spearman_r = {}
    spearman_p = {}
    sig_flags  = {}

    for idx in valid_topo:
        pearson_r[idx]  = {}
        pearson_p[idx]  = {}
        spearman_r[idx] = {}
        spearman_p[idx] = {}
        sig_flags[idx]  = {}
        x = clean_df[idx].values.astype(np.float64)
        for prop in available_targets:
            y = clean_df[prop].values.astype(np.float64)
            try:
                pr, pp = pearsonr(x, y)
                sr, sp = spearmanr(x, y)
            except Exception:
                pr = pp = sr = sp = float('nan')
            pearson_r[idx][prop]  = round(float(pr), 4) if not math.isnan(float(pr)) else 0.0
            pearson_p[idx][prop]  = round(float(pp), 4) if not math.isnan(float(pp)) else 1.0
            spearman_r[idx][prop] = round(float(sr), 4) if not math.isnan(float(sr)) else 0.0
            spearman_p[idx][prop] = round(float(sp), 4) if not math.isnan(float(sp)) else 1.0
            pv = float(pp) if not math.isnan(float(pp)) else 1.0
            sig_flags[idx][prop] = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"

    # VIF
    vif_scores = {}
    topo_arr = clean_df[valid_topo].values.astype(np.float64)
    for i, col in enumerate(valid_topo):
        others_idx = [j for j in range(len(valid_topo)) if j != i]
        if not others_idx:
            vif_scores[col] = 1.0
            continue
        X_oth = topo_arr[:, others_idx]
        y_col = topo_arr[:, i]
        try:
            from sklearn.linear_model import LinearRegression as _LR
            r2 = _LR().fit(X_oth, y_col).score(X_oth, y_col)
            vif_scores[col] = round(1.0 / (1.0 - r2) if r2 < 0.9999 else 999.0, 3)
        except Exception:
            vif_scores[col] = None

    print("    [\u2713] Pearson + Spearman + VIF computed.")
    return {
        "pearson":      pearson_r,
        "pearson_p":    pearson_p,
        "spearman":     spearman_r,
        "spearman_p":   spearman_p,
        "significance": sig_flags,
        "vif":          vif_scores,
    }


# ==========================================
# ML MODELS
# ==========================================
MODELS_AND_GRIDS = {
    "LinearReg": (LinearRegression(), {}),
    "Ridge": (
        Ridge(random_state=RANDOM_STATE),
        {"model__alpha": [0.1, 1.0, 10.0]},
    ),
    "Lasso": (
        Lasso(random_state=RANDOM_STATE, max_iter=3000),
        {"model__alpha": [0.01, 0.1, 1.0]},
    ),
    "ElasticNet": (
        ElasticNet(random_state=RANDOM_STATE, max_iter=3000),
        {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.3, 0.5, 0.7]},
    ),
    "RandomForest": (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20]},
    ),
    "ExtraTrees": (
        ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]},
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 5]},
    ),
    "XGBoost": (
        XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=-1),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 6]},
    ),
    "SVR": (
        SVR(),
        {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["rbf", "linear"], "model__gamma": ["scale", "auto"]},
    ),
    "GaussianProcess": (
        GaussianProcessRegressor(
            kernel=Matern(nu=1.5) + WhiteKernel(),
            random_state=RANDOM_STATE, normalize_y=True
        ),
        {},
    ),
    "NeuralNet": (
        MLPRegressor(max_iter=1000, random_state=RANDOM_STATE),
        {"model__hidden_layer_sizes": [(64, 32), (128, 64)], "model__alpha": [0.0001, 0.001]},
    ),
}


# ==========================================
# ALGORITHM 3: ML with k-fold CV
# ==========================================
def run_ml_qspr(df: pd.DataFrame):
    print(f"[*] Running ML with {CV_FOLDS}-fold CV...")

    # Build clean float64 feature matrix
    valid_topo = [t for t in TOPO_INDICES if t in df.columns]
    clean_topo = _sanitise_df(df, valid_topo)
    if len(clean_topo) < 10:
        print("    [!] Insufficient clean data for ML.")
        return pd.DataFrame(), {}

    X = clean_topo[valid_topo].values.astype(np.float64)
    valid_idx = clean_topo.index

    # Valid targets: numeric, >50% filled
    available_targets = []
    for t in ML_TARGETS:
        if t not in df.columns:
            continue
        series = pd.Series([_safe_float(v) for v in df.loc[valid_idx, t]])
        if series.notna().sum() >= max(10, len(valid_idx) * 0.5):
            available_targets.append(t)

    print(f"    [*] Predicting {len(available_targets)} properties: {available_targets}")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_results = []
    best_models = {}

    for prop in available_targets:
        y_raw = pd.Series([_safe_float(v) for v in df.loc[valid_idx, prop]])
        mask = y_raw.notna().values
        X_prop = X[mask]
        y = y_raw.values[mask].astype(np.float64)

        if len(y) < 10:
            print(f"    [!] Skipping {prop}: only {len(y)} valid rows.")
            continue

        best_r2 = -np.inf
        best_model_for_prop = None

        for name, (model, param_grid) in MODELS_AND_GRIDS.items():
            try:
                pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])

                if param_grid:
                    gs = GridSearchCV(
                        pipe, param_grid,
                        cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
                        scoring="r2", n_jobs=-1, error_score="raise"
                    )
                    gs.fit(X_prop, y)
                    tuned_pipe = gs.best_estimator_
                else:
                    tuned_pipe = pipe

                cv_r2  = cross_val_score(tuned_pipe, X_prop, y, cv=kf, scoring="r2")
                cv_mae = cross_val_score(tuned_pipe, X_prop, y, cv=kf,
                                         scoring="neg_mean_absolute_error")

                mean_r2  = float(np.mean(cv_r2))
                std_r2   = float(np.std(cv_r2))
                mean_mae = float(-np.mean(cv_mae))

                all_results.append({
                    "Property": prop, "Model": name,
                    "R2_mean":  round(mean_r2,  4),
                    "R2_std":   round(std_r2,   4),
                    "MAE_mean": round(mean_mae,  4),
                })

                if mean_r2 > best_r2:
                    best_r2 = mean_r2
                    tuned_pipe.fit(X_prop, y)
                    best_model_for_prop = (tuned_pipe, valid_idx[mask])

            except Exception as e:
                print(f"    [!] {name} failed for {prop}: {e}")
                continue

        if best_model_for_prop is not None:
            best_models[prop] = best_model_for_prop
            print(f"    [\u2713] {prop}: best R\u00b2={best_r2:.3f}")

    return pd.DataFrame(all_results), best_models


# ==========================================
# SHAP Feature Importance
# ==========================================
def compute_shap(df: pd.DataFrame, best_models: dict) -> dict:
    print("[*] Computing SHAP feature importance...")
    valid_topo = [t for t in TOPO_INDICES if t in df.columns]
    shap_summaries = {}

    for prop, model_info in best_models.items():
        if model_info is None:
            continue
        try:
            pipe, row_idx = model_info
            inner   = pipe.named_steps["model"]
            scaler  = pipe.named_steps["scaler"]

            # Build X for the rows used in training
            X_sub = _sanitise_df(df.loc[row_idx], valid_topo)
            if len(X_sub) == 0:
                continue
            X_arr     = X_sub[valid_topo].values.astype(np.float64)
            X_scaled  = scaler.transform(X_arr)

            if isinstance(inner, (RandomForestRegressor, XGBRegressor,
                                   ExtraTreesRegressor, GradientBoostingRegressor)):
                explainer = shap.TreeExplainer(inner)
                shap_vals = explainer.shap_values(X_scaled)
            else:
                bg = shap.kmeans(X_scaled, min(10, len(X_scaled)))
                explainer = shap.KernelExplainer(inner.predict, bg)
                shap_vals = explainer.shap_values(X_scaled, nsamples=50)

            mean_abs = np.abs(np.array(shap_vals)).mean(axis=0)
            shap_summaries[prop] = {
                feat: round(float(val), 6)
                for feat, val in zip(valid_topo, mean_abs)
            }
            print(f"    [\u2713] SHAP done for {prop}")
        except Exception as e:
            print(f"    [!] SHAP failed for {prop}: {e}")

    return shap_summaries


# ==========================================
# DRUG-LIKENESS FILTERS
# ==========================================
def apply_drug_filters(df: pd.DataFrame) -> pd.DataFrame:
    print("[*] Applying drug-likeness filters...")

    def _safe_cmp(series, threshold, op="le"):
        nums = pd.Series([_safe_float(v) for v in series])
        if op == "le":
            return nums <= threshold
        return nums >= threshold

    df["Lipinski_Pass"] = (
        _safe_cmp(df.get("MolWt",   pd.Series([999]*len(df))), 500) &
        _safe_cmp(df.get("LogP",    pd.Series([999]*len(df))), 5)   &
        _safe_cmp(df.get("HBD",     pd.Series([999]*len(df))), 5)   &
        _safe_cmp(df.get("HBA",     pd.Series([999]*len(df))), 10)
    )

    df["Veber_Pass"] = (
        _safe_cmp(df.get("RotBonds", pd.Series([999]*len(df))), 10)  &
        _safe_cmp(df.get("TPSA",     pd.Series([999]*len(df))), 140)
    )

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    pains_flags = []
    for smiles in df["SMILES"]:
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            pains_flags.append(not catalog.HasMatch(mol))
        except Exception:
            pains_flags.append(None)
    df["PAINS_Pass"] = pains_flags

    lip = df["Lipinski_Pass"].sum()
    veb = df["Veber_Pass"].sum()
    pai = pd.Series(pains_flags).eq(True).sum()
    total = len(df)
    print(f"    [\u2713] Lipinski: {lip}/{total} | Veber: {veb}/{total} | PAINS: {pai}/{total}")
    return df
