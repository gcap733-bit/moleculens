# ==========================================
# main.py — pipeline orchestrator
# Run: python main.py --disease diabetes
# ==========================================

import os
import argparse
import pandas as pd

from config import OUTPUT_DIR, MAX_DRUGS
from validator import validate_disease_input
from fetcher import fetch_all
from ml import (
    compute_topological_indices,
    run_correlation,
    run_ml_qspr,
    compute_shap,
    apply_drug_filters,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_pipeline(disease: str, max_drugs: int = MAX_DRUGS) -> dict:
    """
    Full pipeline: fetch → compute → filter → ML → SHAP.
    Returns a results dict suitable for JSON serialisation
    (used by both CLI and the FastAPI website).
    """
    print(f"\n{'='*50}")
    print(f"  Drug Pipeline: {disease.upper()}")
    print(f"{'='*50}\n")

    # 1. Fetch all data
    df = fetch_all(disease, max_drugs)
    if df.empty:
        return {"error": f"No data found for '{disease}'."}

    # 2. Drug-likeness filters
    df = apply_drug_filters(df)

    # 3. Topological indices
    df = compute_topological_indices(df)

    # 4. Pearson correlation
    corr = run_correlation(df)

    # 5. ML with CV + tuning
    ml_results, best_models = run_ml_qspr(df)

    # 6. SHAP
    shap_summaries = compute_shap(df, best_models)

    # 7. Best model per property
    top_models = (
        ml_results.sort_values("R2_mean", ascending=False)
        .drop_duplicates("Property")
        .reset_index(drop=True)
    )

    # 8. Export CSV
    out_path = os.path.join(OUTPUT_DIR, f"{disease}_results.csv")
    df.to_csv(out_path, index=False)

    print(f"\n[*] Pipeline complete → {out_path}")
    print(f"    Drugs processed : {len(df)}")
    print(f"    Total columns   : {len(df.columns)}")
    print("\n--- Best model per property ---")
    print(top_models.to_string(index=False))

    return {
        "disease":        disease,
        "drug_count":     len(df),
        "columns":        list(df.columns),
        "correlation":    corr.round(3).to_dict(),
        "ml_results":     ml_results.to_dict(orient="records"),
        "top_models":     top_models.to_dict(orient="records"),
        "shap":           shap_summaries,
        "lipinski_pass":  int(df["Lipinski_Pass"].sum()),
        "veber_pass":     int(df["Veber_Pass"].sum()),
        "pains_pass":     int(df["PAINS_Pass"].eq(True).sum()),
        "csv_path":       out_path,
        "drugs_preview":  df.head(10).to_dict(orient="records"),
    }


def main():
    parser = argparse.ArgumentParser(description="Drug QSPR Pipeline")
    parser.add_argument("--disease",   type=str, required=True, help="Disease name (e.g. diabetes)")
    parser.add_argument("--max_drugs", type=int, default=MAX_DRUGS, help="Max drugs to fetch")
    parser.add_argument("--no_cache",  action="store_true", help="Bypass disk cache")
    args = parser.parse_args()

    if args.no_cache:
        import config
        config.CACHE_ENABLED = False

    # Validate
    validation = validate_disease_input(args.disease)
    if not validation["valid"]:
        for e in validation["errors"]:
            print(f"[!] {e}")
        if validation["suggestions"]:
            print("Did you mean:", ", ".join(f'"{s}"' for s in validation["suggestions"]))
        return

    if validation["warnings"]:
        for w in validation["warnings"]:
            print(f"[!] Warning: {w}")
        confirm = input("Proceed anyway? (y/n): ").strip().lower()
        if confirm != "y":
            return
        validation = validate_disease_input(validation["disease"], force_proceed=True)

    run_pipeline(validation["disease"], args.max_drugs)


if __name__ == "__main__":
    main()
