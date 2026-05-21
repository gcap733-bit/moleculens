# ==========================================
# INSTALL: pip install rapidfuzz
# ==========================================

import re
import time
import requests
from rapidfuzz import process, fuzz
from chembl_webresource_client.new_client import new_client

# ==========================================
# CURATED MeSH DISEASE LIST
# Covers the most common ChEMBL-indexed conditions.
# Extend this list freely — the fuzzy matcher handles
# misspellings and word-order variation automatically.
# ==========================================
KNOWN_DISEASES = [
    "diabetes", "type 2 diabetes", "type 1 diabetes", "diabetes mellitus",
    "hypertension", "high blood pressure",
    "cancer", "breast cancer", "lung cancer", "colorectal cancer",
    "prostate cancer", "leukemia", "lymphoma", "melanoma",
    "asthma", "chronic obstructive pulmonary disease", "copd",
    "alzheimer", "alzheimer disease", "alzheimer's disease",
    "parkinson", "parkinson disease", "parkinson's disease",
    "epilepsy", "depression", "schizophrenia", "bipolar disorder",
    "anxiety", "obsessive compulsive disorder",
    "rheumatoid arthritis", "osteoarthritis", "osteoporosis",
    "cardiovascular disease", "coronary artery disease", "heart failure",
    "atrial fibrillation", "stroke", "atherosclerosis",
    "hiv", "tuberculosis", "malaria", "hepatitis", "hepatitis b", "hepatitis c",
    "influenza", "covid", "covid-19",
    "kidney disease", "chronic kidney disease", "renal failure",
    "liver disease", "cirrhosis", "fatty liver",
    "hypothyroidism", "hyperthyroidism", "thyroid",
    "obesity", "metabolic syndrome",
    "migraine", "multiple sclerosis",
    "psoriasis", "eczema", "atopic dermatitis",
    "inflammatory bowel disease", "crohn", "ulcerative colitis",
    "anemia", "sickle cell anemia",
    "gout", "lupus", "fibromyalgia",
    "glaucoma", "macular degeneration",
]


# ==========================================
# LAYER 1: Sanitize
# ==========================================
def sanitize_input(raw: str) -> tuple[bool, str, str]:
    """
    Cleans and validates the raw string.
    Returns (is_valid, cleaned_string, error_message).
    """
    if not isinstance(raw, str):
        return False, "", "Input must be a text string."

    cleaned = raw.strip().lower()
    cleaned = re.sub(r'\s+', ' ', cleaned)         # collapse multiple spaces
    cleaned = re.sub(r"[''`]", "'", cleaned)        # normalize apostrophes
    cleaned = re.sub(r'[–—]', '-', cleaned)         # normalize dashes

    if len(cleaned) < 2:
        return False, "", "Input is too short. Please enter a disease name."

    if len(cleaned) > 60:
        return False, "", "Input is too long. Please enter a specific disease name (max 60 characters)."

    if not re.match(r"^[a-z0-9\s\-'\.]+$", cleaned):
        return False, "", "Input contains invalid characters. Use only letters, numbers, spaces, hyphens, or apostrophes."

    if re.match(r'^[\d\s]+$', cleaned):
        return False, "", "Input appears to be numeric. Please enter a disease name."

    return True, cleaned, ""


# ==========================================
# LAYER 2: Fuzzy Match Against Known Disease List
# ==========================================
def fuzzy_match(cleaned: str, top_n: int = 3) -> dict:
    """
    Matches the cleaned input against KNOWN_DISEASES using
    rapidfuzz token_sort_ratio (handles word-order variation,
    e.g. 'mellitus diabetes' still matches 'diabetes mellitus').

    Returns a dict with:
      - status: 'accepted' | 'suggestions' | 'rejected'
      - matched: best matching disease name (if accepted)
      - suggestions: list of close matches (if suggestions)
      - score: best match score
      - message: human-readable message
    """
    results = process.extract(
        cleaned,
        KNOWN_DISEASES,
        scorer=fuzz.token_sort_ratio,
        limit=top_n
    )
    # results: list of (match, score, index)

    best_match, best_score, _ = results[0]

    if best_score >= 90:
        return {
            "status": "accepted",
            "matched": best_match,
            "suggestions": [],
            "score": best_score,
            "message": None,
        }
    elif best_score >= 60:
        suggestions = [r[0] for r in results if r[1] >= 60]
        return {
            "status": "suggestions",
            "matched": None,
            "suggestions": suggestions,
            "score": best_score,
            "message": (
                f"'{cleaned}' wasn't recognised. Did you mean: "
                + ", ".join(f'"{s}"' for s in suggestions) + "?"
            ),
        }
    else:
        return {
            "status": "rejected",
            "matched": None,
            "suggestions": [],
            "score": best_score,
            "message": (
                f"'{cleaned}' doesn't match any known disease in our database. "
                "Please check your spelling or try a broader term (e.g. 'cancer', 'diabetes')."
            ),
        }


# ==========================================
# LAYER 3: Live ChEMBL Confirmation
# ==========================================
def confirm_with_chembl(disease_name: str, min_drugs: int = 10) -> dict:
    """
    Probes ChEMBL to check how many drugs are indexed for this disease
    BEFORE running the full pipeline. Avoids wasting time on diseases
    with no useful data.

    Returns a dict with:
      - status: 'ok' | 'warn' | 'blocked'
      - drug_count: number of drugs found
      - message: human-readable message
    """
    try:
        indication_api = new_client.drug_indication
        # Collect IDs up to a cap (500) to avoid iterating thousands of pages
        ids = set()
        qs = indication_api.filter(
            mesh_heading__icontains=disease_name
        ).only(['molecule_chembl_id'])
        for ind in qs:
            ids.add(ind['molecule_chembl_id'])
            if len(ids) >= 500:
                break
        count = len(ids)

        if count == 0:
            return {
                "status": "blocked",
                "drug_count": 0,
                "message": (
                    f"No drugs found in ChEMBL for '{disease_name}'. "
                    "Try a broader or alternative disease name."
                ),
            }
        elif count < min_drugs:
            return {
                "status": "warn",
                "drug_count": count,
                "message": (
                    f"Only {count} drug(s) found for '{disease_name}' in ChEMBL. "
                    "Results may not be statistically meaningful. Proceed anyway?"
                ),
            }
        else:
            return {
                "status": "ok",
                "drug_count": count,
                "message": f"Found {count} drugs for '{disease_name}'. Ready to run.",
            }
    except Exception as e:
        return {
            "status": "warn",
            "drug_count": -1,
            "message": f"Could not verify with ChEMBL right now ({e}). Proceeding with caution.",
        }


# ==========================================
# MAIN VALIDATOR — combines all 3 layers
# Used by both the Python script and the website API
# ==========================================
def validate_disease_input(
    raw_input: str,
    skip_chembl_check: bool = False,
    force_proceed: bool = False,
) -> dict:
    """
    Full 3-layer validation pipeline.

    Args:
        raw_input:          The raw disease string from the user.
        skip_chembl_check:  If True, skip the live ChEMBL probe (faster, offline).
        force_proceed:      If True, allow pipeline to run even on 'warn' ChEMBL status.

    Returns a dict:
        {
          "valid": bool,           # True = pipeline can run
          "disease": str | None,   # Cleaned, confirmed disease name
          "drug_count": int,       # -1 if check skipped
          "suggestions": list,     # Non-empty if fuzzy match found alternatives
          "warnings": list,        # Non-fatal issues to surface in the UI
          "errors": list,          # Fatal issues — pipeline must not run
        }
    """
    result = {
        "valid": False,
        "disease": None,
        "drug_count": -1,
        "suggestions": [],
        "warnings": [],
        "errors": [],
    }

    # --- Layer 1: Sanitize ---
    is_valid, cleaned, error_msg = sanitize_input(raw_input)
    if not is_valid:
        result["errors"].append(error_msg)
        return result

    # --- Layer 2: Fuzzy match ---
    match = fuzzy_match(cleaned)

    if match["status"] == "rejected":
        result["errors"].append(match["message"])
        return result

    if match["status"] == "suggestions":
        result["suggestions"] = match["suggestions"]
        result["errors"].append(match["message"])
        return result

    # Accepted — use the canonical matched name
    canonical = match["matched"]

    # --- Layer 3: ChEMBL live check ---
    if not skip_chembl_check:
        chembl = confirm_with_chembl(canonical)
        result["drug_count"] = chembl["drug_count"]

        if chembl["status"] == "blocked":
            result["errors"].append(chembl["message"])
            return result

        if chembl["status"] == "warn":
            result["warnings"].append(chembl["message"])
            if not force_proceed:
                # Surface warning to user — let them decide
                result["suggestions"] = []
                result["disease"] = canonical
                return result  # valid=False still, awaiting user confirmation

    result["valid"] = True
    result["disease"] = canonical
    return result


# ==========================================
# USAGE IN PYTHON SCRIPT
# ==========================================
def validated_extract(raw_disease_input: str, max_drugs: int = 100):
    """
    Drop-in replacement for extract_disease_data() with validation.
    Shows suggestions interactively in the terminal.
    """
    print(f"[*] Validating input: '{raw_disease_input}'")
    validation = validate_disease_input(raw_disease_input)

    if validation["errors"]:
        for e in validation["errors"]:
            print(f"[!] {e}")
        if validation["suggestions"]:
            print("\nSuggestions:")
            for i, s in enumerate(validation["suggestions"], 1):
                print(f"  {i}. {s}")
            choice = input("\nEnter number to select, or 0 to cancel: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(validation["suggestions"]):
                chosen = validation["suggestions"][int(choice) - 1]
                return validated_extract(chosen, max_drugs)
        return None

    if validation["warnings"]:
        for w in validation["warnings"]:
            print(f"[!] Warning: {w}")
        confirm = input("Proceed anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            return None
        validation = validate_disease_input(
            validation["disease"], force_proceed=True
        )

    print(f"[✓] Validated: '{validation['disease']}' ({validation['drug_count']} drugs found)")
    return validation["disease"]


# ==========================================
# USAGE AS WEBSITE API ENDPOINT (FastAPI)
# ==========================================
# from fastapi import FastAPI
# app = FastAPI()
#
# @app.post("/validate")
# async def validate_endpoint(body: dict):
#     raw = body.get("disease", "")
#     result = validate_disease_input(raw)
#     return result          # returns JSON directly to the frontend
#
# The frontend then:
#   - if result.valid → proceed with full pipeline call
#   - if result.suggestions → show "Did you mean X, Y, Z?" buttons
#   - if result.errors → show error message inline
#   - if result.warnings → show confirm dialog before proceeding


# ==========================================
# QUICK TEST
# ==========================================
if __name__ == "__main__":
    test_inputs = [
        "diabetes",           # exact match
        "diabtes",            # typo → suggestion
        "type 2 diabeetus",   # typo → suggestion
        "mellitus diabetes",  # word order → should still match
        "xyz123!!",           # garbage → reject
        "hypertension",       # exact match
        "high blod pressure", # typo → suggestion
        "",                   # empty
        "a",                  # too short
    ]

    for inp in test_inputs:
        r = validate_disease_input(inp, skip_chembl_check=True)
        status = "VALID" if r["valid"] else "INVALID"
        disease = r["disease"] or "-"
        errs = r["errors"]
        sugg = r["suggestions"]
        print(f"  [{status}] '{inp}' → disease='{disease}' errors={errs} suggestions={sugg}")
