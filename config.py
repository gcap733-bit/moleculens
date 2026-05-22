# ==========================================
# config.py — all settings in one place
# Change values here; nothing else needs editing
# ==========================================

import os

# --- Pipeline ---
MAX_DRUGS = 200          # max drugs to fetch per disease (increased from 100)
MIN_DRUGS_WARN = 10      # warn user if fewer drugs found
MIN_DRUGS_ERROR = 5      # ERROR if fewer than this (was allowing 0)
TEST_SIZE = 0.2          # train/test split ratio
CV_FOLDS = 5             # k-fold cross-validation folds
RANDOM_STATE = 42

# --- API rate limits ---
PUBCHEM_DELAY = 0.2      # seconds between PubChem requests (max 5/sec)
PKCMS_DELAY   = 0.5      # seconds between pkCSM requests
MAX_RETRIES   = 3        # retry attempts on network failure
RETRY_BACKOFF = 2.0      # exponential backoff multiplier

# --- Caching ---
# On Vercel the project filesystem is read-only; /tmp is the only writable area.
_ON_VERCEL = os.environ.get("VERCEL") == "1"
_BASE_DIR  = "/tmp" if _ON_VERCEL else os.path.dirname(__file__)
CACHE_DIR  = os.path.join(_BASE_DIR, ".cache")
CACHE_ENABLED = True

# --- Output ---
OUTPUT_DIR = os.path.join(_BASE_DIR, "outputs")

# --- FastAPI ---
API_HOST = "0.0.0.0"
API_PORT = int(os.environ.get("PORT", 5000))
CORS_ORIGINS = ["*"]     # restrict this in production

# --- Topological indices computed: 39 Original + 11 New = 50 Total ---
TOPO_INDICES = [
    # ── Original degree-based (9) ──────────────────────
    "M1",   # First Zagreb index
    "M2",   # Second Zagreb index
    "ABC",  # Atom-bond connectivity
    "R",    # Randic connectivity
    "H",    # Harmonic index
    "F",    # Forgotten topological index
    "AZI",  # Augmented Zagreb index
    "GA",   # Geometric-Arithmetic index
    "SC",   # Sum-Connectivity index
    # ── New degree-based from image (11 new) ───────────
    "BM",   # Bi-Zagreb: sum(u+v+uv)
    "TM",   # Tri-Zagreb: sum(u²+v²+uv)
    "GH",   # Geometric-Harmonic: sum(sqrt(uv)(u+v)/2)
    "GBM",  # Geometric Bi-Zagreb: sum(sqrt(uv)/(u+v+uv))
    "GTM",  # Geometric Tri-Zagreb: sum(sqrt(uv)/(u²+v²+uv))
    "HG",   # Harmonic-Geometric: sum(2/(sqrt(uv)(u+v)))
    "BMG",  # Bi Zagreb-Geometric: sum((u+v+uv)/sqrt(uv))
    "BMH",  # Bi Zagreb-Harmonic: sum((u+v+uv)(u+v)/2)
    "TMG",  # Tri Zagreb-Geometric: sum((u²+v²+uv)/sqrt(uv))
    "TMH",  # Tri Zagreb-Harmonic: sum((u²+v²+uv)(u+v)/2)
    "SDD",  # Symmetric Degree Division: sum((u²+v²)/uv)
    # ── Reverse-degree variants (use n+1-d as degree) ──
    "RM1",  # Reverse First Zagreb
    "RM2",  # Reverse Second Zagreb
    "RABC", # Reverse ABC
    "RR",   # Reverse Randic
    "RH",   # Reverse Harmonic
    "RF",   # Reverse Forgotten
    "RGA",  # Reverse Geometric-Arithmetic
    "RBM",  # Reverse Bi-Zagreb
    "RSDD", # Reverse Symmetric Degree Division
    # ── Degree-sum variants (use d(u)+d(v) as weight) ──
    "DS1",  # Degree-Sum Zagreb 1: sum((u+v)²)
    "DS2",  # Degree-Sum Zagreb 2: sum((u+v)(u+v))
    "DSR",  # Degree-Sum Randic: sum(1/sqrt(u+v))
    "DSH",  # Degree-Sum Harmonic: sum(2/(u+v))
    "DSGA", # Degree-Sum GA: sum(2sqrt(uv)/(u+v))
    # ── Original Distance-based (5) ────────────────────
    "W",    # Wiener index (original)
    "J",    # Balaban J index
    "Z",    # Hosoya Z index
    "Sz",   # Szeged index (original)
    "GE",   # Graph entropy (Shannon)
    # ── NEW: Advanced distance-based Wiener variants (6) 
    "W_v",     # Vertex Wiener: sum over pairs of vertices
    "W_e",     # Edge Wiener: sum over pairs of edges
    "W_ve",    # Vertex-edge Wiener: mixed
    "Sz_v",    # Vertex Szeged: based on vertex partitioning
    "Sz_e",    # Edge Szeged: based on edge partitioning
    "Sz_ve",   # Vertex-edge Szeged: mixed
    # ── NEW: Mostar indices (2)
    "Mo_v",    # Vertex Mostar
    "Mo_e",    # Edge Mostar
    # ── NEW: Special indices (3)
    "PI",      # Padmakar-Ivan index
    "Schultz", # Schultz index
    "Gutman",  # Gutman index
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

