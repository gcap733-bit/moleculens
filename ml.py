# ==========================================
# ml.py — topological indices, ML, SHAP, filters
# Updated: 135 topological indices (50 original + 85 new)
#   New additions from:
#     degreessum_computation.py           → 30 SS_ neighbour-degree-sum variants
#     degree_reverse_computation.py       → 11 new degree-based + 44 Rk reverse variants
# ==========================================

import math
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
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
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return float('nan')
        return f
    except Exception:
        return float('nan')


def _df_to_float_array(df, cols):
    arr = np.zeros((len(df), len(cols)), dtype=np.float64)
    for j, col in enumerate(cols):
        series = df[col]
        for i, val in enumerate(series):
            arr[i, j] = _safe_float(val)
    return arr


def _sanitise_df(df, cols):
    out = pd.DataFrame(index=df.index)
    for col in cols:
        out[col] = [_safe_float(v) for v in df[col]]
    out = out.dropna()
    return out


# ==========================================
# HELPER: k-th reverse degree (from degree_reverse_computation.py)
# R_k(v) = Δ - d(v) + k          when k ≤ d(v)
#           (Δ - d(v) + k) mod Δ  otherwise
# ==========================================
def _rev_deg_k(d_val: int, k: int, delta: int) -> int:
    if delta == 0:
        return 0
    if k <= d_val:
        return delta - d_val + k
    else:
        return (delta - d_val + k) % delta


# ==========================================
# DISTANCE MATRIX — uses RDKit C++ implementation (fast)
# ==========================================
def _get_distance_matrix(mol):
    n = mol.GetNumAtoms()
    try:
        dm = rdmolops.GetDistanceMatrix(mol)
        INF = float('inf')
        result = []
        for i in range(n):
            row = []
            for j in range(n):
                val = dm[i][j]
                row.append(INF if val >= 1e7 else float(val))
            result.append(row)
        return result
    except Exception:
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
# HOSOYA Z INDEX — iterative DP
# ==========================================
def _hosoya_z(edge_list, n_atoms):
    m = len(edge_list)
    if m == 0:
        return 1.0
    if m > 25:
        return float(1 + m + m * (m - 1) // 4)
    p = [0] * (m // 2 + 2)
    p[0] = 1

    def _count(idx, used):
        if idx == m:
            return
        _count(idx + 1, used)
        u, v = edge_list[idx]
        if not used[u] and not used[v]:
            used[u] = used[v] = True
            sz = sum(used) // 2
            if sz < len(p):
                p[sz] += 1
            _count(idx + 1, used)
            used[u] = used[v] = False

    if m <= 18:
        used = [False] * n_atoms
        _count(0, used)
    return float(sum(p))


# ==========================================
# ALGORITHM 1: 135 Topological Indices
# ==========================================
def compute_topological_indices(df: pd.DataFrame) -> pd.DataFrame:
    print("[*] Computing 135 topological indices...")
    all_keys = TOPO_INDICES
    results = {key: [] for key in all_keys}

    for smiles in df["SMILES"]:
        try:
            mol = Chem.RemoveHs(Chem.MolFromSmiles(str(smiles)))
            if mol is None:
                raise ValueError("Invalid SMILES")
            n       = mol.GetNumAtoms()
            m_bonds = mol.GetNumBonds()
            INF     = float("inf")

            degrees = [a.GetDegree() for a in mol.GetAtoms()]
            deg_sum = sum(degrees)
            delta   = max(degrees) if degrees else 0   # max degree Δ (for Rk)

            # ──────────────────────────────────────────────
            # BLOCK 1 — Original degree-based (9)
            # ──────────────────────────────────────────────
            m1 = m2 = abc = r = h = f = azi = ga = sc = 0.0
            for d in degrees:
                m1 += d * d
                f  += d * d * d

            for bond in mol.GetBonds():
                u = bond.GetBeginAtom().GetDegree()
                v = bond.GetEndAtom().GetDegree()
                if u == 0 or v == 0:
                    continue
                uv    = u * v
                uv_s  = u + v
                m2   += uv
                r    += 1.0 / math.sqrt(uv)
                h    += 2.0 / uv_s
                sc   += 1.0 / math.sqrt(uv_s)
                ga   += 2.0 * math.sqrt(uv) / uv_s
                if uv_s - 2 > 0:
                    abc += math.sqrt((uv_s - 2) / uv)
                if uv_s - 2 != 0:
                    azi += (uv / (uv_s - 2)) ** 3

            # ──────────────────────────────────────────────
            # BLOCK 2 — New degree-based (11)
            # ──────────────────────────────────────────────
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

            # ──────────────────────────────────────────────
            # BLOCK 3 — Reverse-degree (9)  rd(v) = n+1-d(v)
            # ──────────────────────────────────────────────
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

            # ──────────────────────────────────────────────
            # BLOCK 4 — Degree-sum edge variants (5)
            # ──────────────────────────────────────────────
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
                uv_p = u * v
                if uv_p > 0:
                    dsga += 2.0 * math.sqrt(uv_p) / s

            # ──────────────────────────────────────────────
            # BLOCK 5 — NEW: Additional normal degree-based (11)
            # Source: degree_reverse_computation_orderedIndices.py
            # ──────────────────────────────────────────────
            a_idx = g_idx = ha_idx = so_idx = abc_sc_idx = isi_idx = sigma_idx = 0.0
            hbm_idx = htm_idx = bma_idx = tma_idx = 0.0
            for bond in mol.GetBonds():
                u = bond.GetBeginAtom().GetDegree()
                v = bond.GetEndAtom().GetDegree()
                if u == 0 or v == 0:
                    continue
                suv  = float(u + v)
                pruv = float(u * v)
                u2v2 = float(u*u + v*v)

                a_idx     += suv / 2.0
                g_idx     += math.sqrt(pruv)
                ha_idx    += 4.0 / (suv ** 2)
                so_idx    += math.sqrt(u2v2)
                if pruv > 0 and suv > 2:
                    abc_sc_idx += math.sqrt((suv - 2) / pruv) / math.sqrt(suv)
                isi_idx   += pruv / suv
                sigma_idx += float((u - v) ** 2)
                d_hbm = (suv + pruv) * suv
                if d_hbm != 0:
                    hbm_idx += 2.0 / d_hbm
                d_htm = (u2v2 + pruv) * suv
                if d_htm != 0:
                    htm_idx += 2.0 / d_htm
                bma_idx += (2.0 / suv) * (suv + pruv)
                tma_idx += (2.0 / suv) * (u2v2 + pruv)

            # ──────────────────────────────────────────────
            # BLOCK 6 — Graph entropy
            # ──────────────────────────────────────────────
            ge = 0.0
            if deg_sum > 0:
                for d in degrees:
                    if d > 0:
                        p = d / float(deg_sum)
                        ge -= p * math.log2(p)

            # ──────────────────────────────────────────────
            # BLOCK 7 — Distance-based (W, J, Z, Sz, GE)
            # ──────────────────────────────────────────────
            dist = _get_distance_matrix(mol)
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

            edge_list = [
                (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()
            ]
            z_hosoya = _hosoya_z(edge_list, n)

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

            # ──────────────────────────────────────────────
            # BLOCK 8 — Advanced cut-graph indices (11)
            # ──────────────────────────────────────────────
            w_v = float(w)
            bonds = list(mol.GetBonds())
            num_bonds = len(bonds)

            w_e = 0.0
            for idx1 in range(num_bonds):
                for idx2 in range(idx1 + 1, num_bonds):
                    b1 = bonds[idx1]
                    b2 = bonds[idx2]
                    u1, v1 = b1.GetBeginAtomIdx(), b1.GetEndAtomIdx()
                    u2, v2 = b2.GetBeginAtomIdx(), b2.GetEndAtomIdx()
                    d_min = min(
                        dist[u1][u2], dist[u1][v2],
                        dist[v1][u2], dist[v1][v2]
                    )
                    if d_min != INF:
                        w_e += float(d_min + 1)

            w_ve = 0.0
            for i in range(n):
                for bond in bonds:
                    u = bond.GetBeginAtomIdx()
                    v = bond.GetEndAtomIdx()
                    d_val = min(dist[i][u], dist[i][v])
                    if d_val != INF:
                        w_ve += float(d_val)

            sz_v  = float(sz)
            sz_e  = sz_ve = mo_v = mo_e = pi_val = 0.0

            for bond in bonds:
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                n_u = sum(
                    1 for k2 in range(n)
                    if dist[u][k2] != INF and dist[v][k2] != INF
                    and dist[u][k2] < dist[v][k2]
                )
                n_v = sum(
                    1 for k2 in range(n)
                    if dist[u][k2] != INF and dist[v][k2] != INF
                    and dist[v][k2] < dist[u][k2]
                )
                m_u = m_v = 0
                for f_bond in bonds:
                    x = f_bond.GetBeginAtomIdx()
                    y = f_bond.GetEndAtomIdx()
                    if INF not in (dist[x][u], dist[y][u], dist[x][v], dist[y][v]):
                        d_f_u = min(dist[x][u], dist[y][u])
                        d_f_v = min(dist[x][v], dist[y][v])
                        if d_f_u < d_f_v:
                            m_u += 1
                        elif d_f_v < d_f_u:
                            m_v += 1
                sz_e  += float(m_u * m_v)
                sz_ve += 0.5 * (n_u * m_v + n_v * m_u)
                mo_v  += float(abs(n_u - n_v))
                mo_e  += float(abs(m_u - m_v))
                pi_val += float(m_u + m_v)

            sz_ve = float(sz_ve)

            schultz = 0.0
            for i in range(n):
                d_i = degrees[i]
                for j in range(i + 1, n):
                    d_j = degrees[j]
                    d_ij = dist[i][j]
                    if d_ij != INF:
                        schultz += float((d_i + d_j) * d_ij)

            gutman = 0.0
            for i in range(n):
                d_i = degrees[i]
                for j in range(i + 1, n):
                    d_j = degrees[j]
                    d_ij = dist[i][j]
                    if d_ij != INF:
                        gutman += float(d_i * d_j * d_ij)

            # ──────────────────────────────────────────────
            # BLOCK 9 — NEW: Neighbour-degree-sum (SS_) variants (30)
            # Source: degreessum_computation.py
            # σ(v) = Σ_{u~v} d(u)
            # ──────────────────────────────────────────────
            deg_sum_map = {}
            for atom in mol.GetAtoms():
                deg_sum_map[atom.GetIdx()] = sum(
                    nbr.GetDegree() for nbr in atom.GetNeighbors()
                )

            ss_m1 = ss_m2 = ss_bm = ss_tm = ss_sc = ss_gh = ss_r = ss_gbm = 0.0
            ss_a  = ss_g  = ss_ga = ss_h  = ss_hg = ss_hm = ss_hbm = ss_htm = 0.0
            ss_sdd = ss_ha = ss_so = ss_bmg = ss_abc = ss_bmh = ss_az = 0.0
            ss_bma = ss_isi = ss_tmh = ss_abs = ss_tma = ss_sigma = ss_tmg = 0.0

            for bond in mol.GetBonds():
                x = deg_sum_map[bond.GetBeginAtom().GetIdx()]
                y = deg_sum_map[bond.GetEndAtom().GetIdx()]
                if x == 0 or y == 0:
                    continue
                xs   = float(x + y)
                xy   = float(x * y)
                x2y2 = float(x*x + y*y)
                sqxy = math.sqrt(xy)

                ss_m1    += xs
                ss_m2    += xy
                ss_bm    += xs + xy
                ss_tm    += x2y2 + xy
                ss_sc    += 1.0 / math.sqrt(xs)
                ss_gh    += sqxy * xs / 2.0
                ss_r     += 1.0 / sqxy
                d_ss_gbm  = xs + xy
                if d_ss_gbm != 0:
                    ss_gbm += sqxy / d_ss_gbm
                ss_a     += xs / 2.0
                ss_g     += sqxy
                ss_ga    += 2.0 * sqxy / xs
                ss_h     += 2.0 / xs
                d_ss_hg   = sqxy * xs
                if d_ss_hg != 0:
                    ss_hg += 2.0 / d_ss_hg
                ss_hm    += xs ** 2
                d_ss_hbm  = (xs + xy) * xs
                if d_ss_hbm != 0:
                    ss_hbm += 2.0 / d_ss_hbm
                d_ss_htm  = (x2y2 + xy) * xs
                if d_ss_htm != 0:
                    ss_htm += 2.0 / d_ss_htm
                if xy != 0:
                    ss_sdd += x2y2 / xy
                ss_ha    += 4.0 / (xs ** 2)
                ss_so    += math.sqrt(x2y2)
                if sqxy != 0:
                    ss_bmg += (xs + xy) / sqxy
                if xs + xy - 2 > 0 and xy > 0:
                    ss_abc += math.sqrt((xs + xy - 2) / xy)
                    ss_az  += ((xy / (xs + xy - 2)) ** 3)
                    ss_abs += math.sqrt((xs + xy - 2) / xs)
                ss_bmh   += (xs + xy) * xs / 2.0
                ss_bma   += (2.0 / xs) * (xs + xy)
                ss_isi   += xy / xs
                ss_tmh   += (x2y2 + xy) * xs / 2.0
                ss_tma   += (2.0 / xs) * (x2y2 + xy)
                ss_sigma += float((x - y) ** 2)
                if sqxy != 0:
                    ss_tmg += (x2y2 + xy) / sqxy

            # ──────────────────────────────────────────────
            # BLOCK 10 — NEW: Rk reverse variants of new 11 indices (k=1..4)
            # Source: degree_reverse_computation_orderedIndices.py
            # ──────────────────────────────────────────────
            rk_all = {}
            for ki in range(1, 5):      # ki = k, renamed to avoid shadowing loop var
                rk_a = rk_g = rk_ha = rk_so = rk_abc_sc = rk_isi = rk_sigma = 0.0
                rk_hbm = rk_htm = rk_bma = rk_tma = 0.0

                for bond in mol.GetBonds():
                    du = bond.GetBeginAtom().GetDegree()
                    dv = bond.GetEndAtom().GetDegree()
                    if delta == 0:
                        continue
                    ru = _rev_deg_k(du, ki, delta)
                    rv = _rev_deg_k(dv, ki, delta)
                    if ru == 0 or rv == 0:
                        continue
                    rsuv  = float(ru + rv)
                    rpruv = float(ru * rv)
                    ru2v2 = float(ru*ru + rv*rv)

                    rk_a      += rsuv / 2.0
                    rk_g      += math.sqrt(rpruv)
                    rk_ha     += 4.0 / (rsuv ** 2)
                    rk_so     += math.sqrt(ru2v2)
                    if rpruv > 0 and rsuv > 2:
                        rk_abc_sc += math.sqrt((rsuv - 2) / rpruv) / math.sqrt(rsuv)
                    rk_isi    += rpruv / rsuv
                    rk_sigma  += float((ru - rv) ** 2)
                    d_rk_hbm   = (rsuv + rpruv) * rsuv
                    if d_rk_hbm != 0:
                        rk_hbm += 2.0 / d_rk_hbm
                    d_rk_htm   = (ru2v2 + rpruv) * rsuv
                    if d_rk_htm != 0:
                        rk_htm += 2.0 / d_rk_htm
                    rk_bma    += (2.0 / rsuv) * (rsuv + rpruv)
                    rk_tma    += (2.0 / rsuv) * (ru2v2 + rpruv)

                rk_all[f"R{ki}_A"]      = rk_a
                rk_all[f"R{ki}_G"]      = rk_g
                rk_all[f"R{ki}_HA"]     = rk_ha
                rk_all[f"R{ki}_SO"]     = rk_so
                rk_all[f"R{ki}_ABC_SC"] = rk_abc_sc
                rk_all[f"R{ki}_ISI"]    = rk_isi
                rk_all[f"R{ki}_sigma"]  = rk_sigma
                rk_all[f"R{ki}_HBM"]    = rk_hbm
                rk_all[f"R{ki}_HTM"]    = rk_htm
                rk_all[f"R{ki}_BMA"]    = rk_bma
                rk_all[f"R{ki}_TMA"]    = rk_tma

            # ──────────────────────────────────────────────
            # Assemble val_map for all 135 indices
            # ──────────────────────────────────────────────
            val_map = {
                # ── Original degree-based (9) ──
                "M1": m1, "M2": m2, "ABC": abc, "R": r, "H": h,
                "F": f, "AZI": azi, "GA": ga, "SC": sc,
                # ── New degree-based (11) ──
                "BM": bm, "TM": tm, "GH": gh, "GBM": gbm, "GTM": gtm,
                "HG": hg, "BMG": bmg, "BMH": bmh, "TMG": tmg, "TMH": tmh,
                "SDD": sdd,
                # ── Reverse-degree (9) ──
                "RM1": rm1, "RM2": rm2, "RABC": rabc, "RR": rr, "RH": rh,
                "RF": rf, "RGA": rga, "RBM": rbm, "RSDD": rsdd,
                # ── Degree-sum edge (5) ──
                "DS1": ds1, "DS2": ds2, "DSR": dsr, "DSH": dsh, "DSGA": dsga,
                # ── Distance-based (5) ──
                "W": w, "J": j_bal, "Z": z_hosoya, "Sz": sz, "GE": ge,
                # ── Advanced cut-graph (11) ──
                "W_v": w_v, "W_e": w_e, "W_ve": w_ve,
                "Sz_v": sz_v, "Sz_e": sz_e, "Sz_ve": sz_ve,
                "Mo_v": mo_v, "Mo_e": mo_e,
                "PI": pi_val, "Schultz": schultz, "Gutman": gutman,

                # ══ NEW INDICES (85) ══════════════════════════

                # ── New normal degree-based (11) ──
                "A":      a_idx,
                "G":      g_idx,
                "HA":     ha_idx,
                "SO":     so_idx,
                "ABC_SC": abc_sc_idx,
                "ISI":    isi_idx,
                "sigma":  sigma_idx,
                "HBM":    hbm_idx,
                "HTM":    htm_idx,
                "BMA":    bma_idx,
                "TMA":    tma_idx,

                # ── SS_ neighbour-degree-sum (30) ──
                "SS_M1":    ss_m1,   "SS_M2":    ss_m2,
                "SS_BM":    ss_bm,   "SS_TM":    ss_tm,
                "SS_SC":    ss_sc,   "SS_GH":    ss_gh,
                "SS_R":     ss_r,    "SS_GBM":   ss_gbm,
                "SS_A":     ss_a,    "SS_G":     ss_g,
                "SS_GA":    ss_ga,   "SS_H":     ss_h,
                "SS_HG":    ss_hg,   "SS_HM":    ss_hm,
                "SS_HBM":   ss_hbm,  "SS_HTM":   ss_htm,
                "SS_SDD":   ss_sdd,  "SS_HA":    ss_ha,
                "SS_SO":    ss_so,   "SS_BMG":   ss_bmg,
                "SS_ABC":   ss_abc,  "SS_BMH":   ss_bmh,
                "SS_AZ":    ss_az,   "SS_BMA":   ss_bma,
                "SS_ISI":   ss_isi,  "SS_TMH":   ss_tmh,
                "SS_ABS":   ss_abs,  "SS_TMA":   ss_tma,
                "SS_sigma": ss_sigma,"SS_TMG":   ss_tmg,

                # ── Rk reverse of new 11 (k=1..4) (44) ──
                **rk_all,
            }

            for key in all_keys:
                results[key].append(_safe_float(val_map.get(key, float('nan'))))

        except Exception:
            for key in all_keys:
                results[key].append(float('nan'))

    expected_len = len(df)
    for key, v_list in results.items():
        if len(v_list) < expected_len:
            v_list = v_list + [float('nan')] * (expected_len - len(v_list))
        elif len(v_list) > expected_len:
            v_list = v_list[:expected_len]
        df[key] = v_list

    # Fill NaN with 0 instead of dropping rows
    for key in all_keys:
        df[key] = df[key].fillna(0.0)

    # Drop only rows where ALL topological indices are zero (completely failed molecules)
    df = df[df[all_keys].any(axis=1)].reset_index(drop=True)

    print(f"    [✓] 135 topological indices computed for {len(df)} molecules.")
    return df


# ==========================================
# ALGORITHM 2: Correlation (Pearson + Spearman + VIF)
# ==========================================
def run_correlation(df: pd.DataFrame) -> dict:
    print("[*] Computing correlation analysis...")

    valid_topo = [t for t in TOPO_INDICES if t in df.columns]
    available_targets = [
        t for t in ML_TARGETS
        if t in df.columns
        and df[t].apply(lambda x: _safe_float(x)).notna().sum() >= max(5, len(df) * 0.3)
    ]

    if not available_targets or not valid_topo:
        print("    [!] No valid targets or indices for correlation.")
        return {"pearson": {}, "pearson_p": {}, "spearman": {}, "spearman_p": {},
                "significance": {}, "vif": {}}

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

    print("    [✓] Pearson + Spearman + VIF computed.")
    return {
        "pearson":      pearson_r,
        "pearson_p":    pearson_p,
        "spearman":     spearman_r,
        "spearman_p":   spearman_p,
        "significance": sig_flags,
        "vif":          vif_scores,
    }


# ==========================================
# ML MODELS — reduced param grids for practical run time
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
        {"model__alpha": [0.1, 1.0], "model__l1_ratio": [0.3, 0.7]},
    ),
    "RandomForest": (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [100], "model__max_depth": [None, 10]},
    ),
    "ExtraTrees": (
        ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [100], "model__max_depth": [None, 10]},
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": [100], "model__learning_rate": [0.1], "model__max_depth": [3, 5]},
    ),
    "XGBoost": (
        XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=-1),
        {"model__n_estimators": [100], "model__learning_rate": [0.1], "model__max_depth": [3, 6]},
    ),
    "SVR": (
        SVR(),
        {"model__C": [1.0, 10.0], "model__kernel": ["rbf"], "model__gamma": ["scale"]},
    ),
    "NeuralNet": (
        MLPRegressor(max_iter=500, random_state=RANDOM_STATE),
        {"model__hidden_layer_sizes": [(64, 32)], "model__alpha": [0.001]},
    ),
}


# ==========================================
# ALGORITHM 3: ML with k-fold CV
# ==========================================
def run_ml_qspr(df: pd.DataFrame):
    print(f"[*] Running ML with {CV_FOLDS}-fold CV...")

    valid_topo = [t for t in TOPO_INDICES if t in df.columns]
    clean_topo = _sanitise_df(df, valid_topo)
    if len(clean_topo) < 10:
        print("    [!] Insufficient clean data for ML.")
        return pd.DataFrame(), {}

    X = clean_topo[valid_topo].values.astype(np.float64)
    valid_idx = clean_topo.index

    available_targets = []
    for t in ML_TARGETS:
        if t not in df.columns:
            continue
        series = pd.Series([_safe_float(v) for v in df.loc[valid_idx, t]])
        if series.notna().sum() >= max(10, len(valid_idx) * 0.5):
            available_targets.append(t)

    print(f"    [*] Predicting {len(available_targets)} properties: {available_targets}")

    all_results = []
    best_models = {}

    for prop in available_targets:
        try:
            y_raw = pd.Series([_safe_float(v) for v in df.loc[valid_idx, prop]])
            mask = y_raw.notna().values
            X_prop = X[mask].copy()
            y = y_raw.values[mask].astype(np.float64)

            if len(y) < 10:
                print(f"    [!] Skipping {prop}: only {len(y)} valid rows.")
                continue

            outer_splits = max(2, min(CV_FOLDS, len(y)))
            inner_splits = max(2, min(3, len(y)))

            kf_outer = KFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
            kf_inner = KFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)

            best_r2 = -np.inf
            best_model_for_prop = None

            for name, (model, param_grid) in MODELS_AND_GRIDS.items():
                try:
                    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
                    if param_grid:
                        gs = GridSearchCV(
                            pipe, param_grid,
                            cv=kf_inner,
                            scoring="r2", n_jobs=-1,
                            error_score=np.nan,
                        )
                        gs.fit(X_prop, y)
                        tuned_pipe = gs.best_estimator_
                    else:
                        tuned_pipe = pipe

                    cv_r2  = cross_val_score(
                        tuned_pipe, X_prop, y, cv=kf_outer,
                        scoring="r2", error_score=np.nan
                    )
                    cv_mae = cross_val_score(
                        tuned_pipe, X_prop, y, cv=kf_outer,
                        scoring="neg_mean_absolute_error",
                        error_score=np.nan,
                    )

                    mean_r2  = float(np.nanmean(cv_r2))
                    std_r2   = float(np.nanstd(cv_r2))
                    mean_mae = float(-np.nanmean(cv_mae))

                    all_results.append({
                        "Property": prop, "Model": name,
                        "R2_mean":  round(mean_r2,  4),
                        "R2_std":   round(std_r2,   4),
                        "MAE_mean": round(mean_mae,  4),
                    })

                    if mean_r2 > best_r2:
                        best_r2 = mean_r2
                        tuned_pipe.fit(X_prop, y)
                        row_indices = list(valid_idx[mask])
                        best_model_for_prop = (tuned_pipe, row_indices)

                except Exception as e:
                    print(f"    [!] {name} failed for {prop}: {e}")
                    continue

            if best_model_for_prop is not None:
                best_models[prop] = best_model_for_prop
                print(f"    [✓] {prop}: best R²={best_r2:.3f}")

        except Exception as e:
            print(f"    [!] Skipping {prop} entirely due to error: {e}")
            continue

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

            X_sub = _sanitise_df(
                df.loc[row_idx].reset_index(drop=True), valid_topo
            )
            if len(X_sub) == 0:
                continue
            X_arr    = X_sub[valid_topo].values.astype(np.float64)
            X_scaled = scaler.transform(X_arr)

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
            print(f"    [✓] SHAP done for {prop}")
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
        _safe_cmp(df.get("MolWt",    pd.Series([999]*len(df))), 500) &
        _safe_cmp(df.get("LogP",     pd.Series([999]*len(df))), 5)   &
        _safe_cmp(df.get("HBD",      pd.Series([999]*len(df))), 5)   &
        _safe_cmp(df.get("HBA",      pd.Series([999]*len(df))), 10)
    )

    df["Veber_Pass"] = (
        _safe_cmp(df.get("RotBonds", pd.Series([999]*len(df))), 10) &
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

    lip   = df["Lipinski_Pass"].sum()
    veb   = df["Veber_Pass"].sum()
    pai   = pd.Series(pains_flags).eq(True).sum()
    total = len(df)
    print(f"    [✓] Lipinski: {lip}/{total} | Veber: {veb}/{total} | PAINS: {pai}/{total}")
    return df
