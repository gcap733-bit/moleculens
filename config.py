# ==========================================
# config.py — all settings in one place
# Change values here; nothing else needs editing
# ==========================================

import os

# --- Pipeline ---
MAX_DRUGS = 100          # max drugs to fetch per disease
MIN_DRUGS_WARN = 10      # warn user if fewer drugs found
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
API_PORT = 8000
CORS_ORIGINS = ["*"]     # restrict this in production

# --- Topological indices computed ---
TOPO_INDICES = ["M1", "M2", "ABC", "R", "H", "F"]

# --- ML targets (ChEMBL-derived properties) ---
ML_TARGETS = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "MolMR"]
