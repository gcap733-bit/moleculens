# ==========================================
# config.py — all settings in one place
# Change values here; nothing else needs editing
# ==========================================

import os

# --- Pipeline ---
MAX_DRUGS = 200          # max drugs to fetch per disease
MIN_DRUGS_WARN = 10      # warn user if fewer drugs found
MIN_DRUGS_ERROR = 5      # ERROR if fewer than this
TEST_SIZE = 0.2          # train/test split ratio
CV_FOLDS = 5             # k-fold cross-validation folds
RANDOM_STATE = 42

# --- API rate limits ---
PUBCHEM_DELAY = 0.2      # seconds between PubChem requests (max 5/sec)
PKCMS_DELAY   = 0.5      # seconds between pkCSM requests
MAX_RETRIES   = 3        # retry attempts on network failure
RETRY_BACKOFF = 2.0      # exponential backoff multiplier

# --- Caching ---
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
CACHE_ENABLED = True

# --- Output ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# --- FastAPI ---
API_HOST = "0.0.0.0"
API_PORT = int(os.environ.get("PORT", 5000))
CORS_ORIGINS = ["*"]     # restrict this in production

# --- Topological indices: 50 original + 85 new = 135 total ---
#
#   Sources merged:
#     degreessum_computation.py      → 30 SS_ neighbour-degree-sum variants
#     degree_reverse_computation.py  → 11 new degree-based + 44 Rk reverse variants
#
TOPO_INDICES = [
    # ════════════════════════════════════════════════════════
    # ORIGINAL 50 INDICES
    # ════════════════════════════════════════════════════════

    # ── Original degree-based (9) ────────────────────────────
    "M1",   # First Zagreb index:       Σ d(v)²
    "M2",   # Second Zagreb index:      Σ d(u)·d(v)
    "ABC",  # Atom-bond connectivity:   Σ √((d(u)+d(v)-2)/(d(u)·d(v)))
    "R",    # Randić connectivity:      Σ 1/√(d(u)·d(v))
    "H",    # Harmonic index:           Σ 2/(d(u)+d(v))
    "F",    # Forgotten index:          Σ d(v)³
    "AZI",  # Augmented Zagreb:         Σ (d(u)·d(v)/(d(u)+d(v)-2))³
    "GA",   # Geometric-Arithmetic:     Σ 2√(d(u)·d(v))/(d(u)+d(v))
    "SC",   # Sum-Connectivity:         Σ 1/√(d(u)+d(v))

    # ── New degree-based (11) ─────────────────────────────────
    "BM",   # Bi-Zagreb:                Σ (d(u)+d(v)+d(u)·d(v))
    "TM",   # Tri-Zagreb:               Σ (d(u)²+d(v)²+d(u)·d(v))
    "GH",   # Geometric-Harmonic:       Σ √(d(u)·d(v))·(d(u)+d(v))/2
    "GBM",  # Geometric Bi-Zagreb:      Σ √(d(u)·d(v))/(d(u)+d(v)+d(u)·d(v))
    "GTM",  # Geometric Tri-Zagreb:     Σ √(d(u)·d(v))/(d(u)²+d(v)²+d(u)·d(v))
    "HG",   # Harmonic-Geometric:       Σ 2/(√(d(u)·d(v))·(d(u)+d(v)))
    "BMG",  # Bi Zagreb-Geometric:      Σ (d(u)+d(v)+d(u)·d(v))/√(d(u)·d(v))
    "BMH",  # Bi Zagreb-Harmonic:       Σ (d(u)+d(v)+d(u)·d(v))·(d(u)+d(v))/2
    "TMG",  # Tri Zagreb-Geometric:     Σ (d(u)²+d(v)²+d(u)·d(v))/√(d(u)·d(v))
    "TMH",  # Tri Zagreb-Harmonic:      Σ (d(u)²+d(v)²+d(u)·d(v))·(d(u)+d(v))/2
    "SDD",  # Sym Degree Division:      Σ (d(u)²+d(v)²)/(d(u)·d(v))

    # ── Reverse-degree (9) — rd(v) = n+1-d(v) ────────────────
    "RM1",  "RM2",  "RABC", "RR",   "RH",
    "RF",   "RGA",  "RBM",  "RSDD",

    # ── Degree-sum edge variants (5) — weight = d(u)+d(v) ────
    "DS1",  "DS2",  "DSR",  "DSH",  "DSGA",

    # ── Distance-based (5) ───────────────────────────────────
    "W",    # Wiener index
    "J",    # Balaban J index
    "Z",    # Hosoya Z index
    "Sz",   # Szeged index
    "GE",   # Graph entropy

    # ── Advanced cut-graph (11) ──────────────────────────────
    "W_v",  "W_e",  "W_ve",
    "Sz_v", "Sz_e", "Sz_ve",
    "Mo_v", "Mo_e",
    "PI",   "Schultz", "Gutman",

    # ════════════════════════════════════════════════════════
    # NEW — 85 ADDITIONAL INDICES FROM UPLOADED FILES
    # ════════════════════════════════════════════════════════

    # ── NEW: Additional normal degree-based (11) ─────────────
    # Source: degree_reverse_computation_orderedIndices.py
    "A",        # Arithmetic index:         Σ (d(u)+d(v))/2
    "G",        # Geometric index:          Σ √(d(u)·d(v))
    "HA",       # Harmonic-square:          Σ 4/(d(u)+d(v))²
    "SO",       # Sombor index:             Σ √(d(u)²+d(v)²)
    "ABC_SC",   # ABC / Sum-Conn product:   Σ √((d(u)+d(v)-2)/(d(u)·d(v))) / √(d(u)+d(v))
    "ISI",      # Inverse Sum Indeg:        Σ d(u)·d(v)/(d(u)+d(v))
    "sigma",    # Sigma irregularity:       Σ (d(u)-d(v))²
    "HBM",      # Harmonic Bi-Zagreb:       Σ 2/((d(u)+d(v)+d(u)·d(v))·(d(u)+d(v)))
    "HTM",      # Harmonic Tri-Zagreb:      Σ 2/((d(u)²+d(v)²+d(u)·d(v))·(d(u)+d(v)))
    "BMA",      # Bi-Zagreb Arithmetic:     Σ (2/(d(u)+d(v)))·(d(u)+d(v)+d(u)·d(v))
    "TMA",      # Tri-Zagreb Arithmetic:    Σ (2/(d(u)+d(v)))·(d(u)²+d(v)²+d(u)·d(v))

    # ── NEW: Neighbour-degree-sum variants, prefix SS_ (30) ──
    # Source: degreessum_computation.py
    # σ(v) = Σ_{u~v} d(u)  (sum of neighbour degrees); replaces d(v) in all formulas
    "SS_M1",   "SS_M2",   "SS_BM",   "SS_TM",   "SS_SC",
    "SS_GH",   "SS_R",    "SS_GBM",  "SS_A",    "SS_G",
    "SS_GA",   "SS_H",    "SS_HG",   "SS_HM",   "SS_HBM",
    "SS_HTM",  "SS_SDD",  "SS_HA",   "SS_SO",   "SS_BMG",
    "SS_ABC",  "SS_BMH",  "SS_AZ",   "SS_BMA",  "SS_ISI",
    "SS_TMH",  "SS_ABS",  "SS_TMA",  "SS_sigma","SS_TMG",

    # ── NEW: Rk reverse-degree variants of new 11 indices (k=1..4) (44) ──
    # Source: degree_reverse_computation_orderedIndices.py
    # R_k(v) = Δ - d(v) + k  when k ≤ d(v);  (Δ - d(v) + k) mod Δ  otherwise
    "R1_A",     "R1_G",     "R1_HA",    "R1_SO",    "R1_ABC_SC",
    "R1_ISI",   "R1_sigma", "R1_HBM",   "R1_HTM",   "R1_BMA",   "R1_TMA",

    "R2_A",     "R2_G",     "R2_HA",    "R2_SO",    "R2_ABC_SC",
    "R2_ISI",   "R2_sigma", "R2_HBM",   "R2_HTM",   "R2_BMA",   "R2_TMA",

    "R3_A",     "R3_G",     "R3_HA",    "R3_SO",    "R3_ABC_SC",
    "R3_ISI",   "R3_sigma", "R3_HBM",   "R3_HTM",   "R3_BMA",   "R3_TMA",

    "R4_A",     "R4_G",     "R4_HA",    "R4_SO",    "R4_ABC_SC",
    "R4_ISI",   "R4_sigma", "R4_HBM",   "R4_HTM",   "R4_BMA",   "R4_TMA",
]

# --- ML targets — all predictable numerical properties ---
ML_TARGETS = [
    # RDKit-computed (original 7)
    "MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "MolMR",
    # PubChem physicochemical (13 numerical)
    "PC_MolecularWeight", "PC_XLogP", "PC_ExactMass", "PC_MonoisotopicMass",
    "PC_TPSA", "PC_Complexity", "PC_FormalCharge", "PC_HBD", "PC_HBA",
    "PC_RotatableBonds", "PC_HeavyAtomCount", "PC_AtomStereoCount",
    "PC_CovalentUnitCount",
    # Bioactivity (4) — only used if data available
    "BIO_IC50", "BIO_Ki", "BIO_EC50", "BIO_Kd",
]
