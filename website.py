# ==========================================
# website.py — FastAPI backend
# FIX: Heavy imports are now lazy (inside functions) so the
#      server starts in <1s and /health always responds fast.
#      Previously rdkit+shap+xgboost imports at module level
#      caused 30-60s startup → Railway health-check killed it.
# Run: uvicorn website:app --host 0.0.0.0 --port $PORT
# ==========================================

import os
import uuid
import math
import json
import re as _re
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

# ── Config only (lightweight — no ML/RDKit imports) ────────────
from config import API_HOST, API_PORT, CORS_ORIGINS, MAX_DRUGS, OUTPUT_DIR

app = FastAPI(
    title="Drug QSPR Pipeline API",
    description="Disease → Drugs → Topological Indices → ML QSPR",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: dict[str, dict] = {}


# ==========================================
# SERVE FRONTEND
# ==========================================
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


# ==========================================
# HEALTH — must respond instantly (no heavy imports here)
# ==========================================
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


# ==========================================
# VALIDATE — lazy import of disease_validator
# ==========================================
class ValidateRequest(BaseModel):
    disease: str


@app.post("/validate")
def validate_endpoint(body: ValidateRequest):
    # Lazy import: chembl_webresource_client only loaded on first call
    from disease_validator import validate_disease_input
    result = validate_disease_input(body.disease)
    return JSONResponse(content=result)


# ==========================================
# RUN — lazy import of pipeline
# ==========================================
class RunRequest(BaseModel):
    disease: str
    max_drugs: Optional[int] = MAX_DRUGS
    force_proceed: Optional[bool] = False


@app.post("/run")
def run_endpoint(body: RunRequest, background_tasks: BackgroundTasks):
    from disease_validator import validate_disease_input
    validation = validate_disease_input(body.disease, force_proceed=body.force_proceed)
    if not validation["valid"]:
        raise HTTPException(status_code=422, detail={
            "errors": validation["errors"],
            "suggestions": validation["suggestions"],
        })

    disease = validation["disease"]
    job_id  = str(uuid.uuid4())[:8]

    JOBS[job_id] = {
        "id":      job_id,
        "disease": disease,
        "status":  "queued",
        "results": None,
        "error":   None,
    }

    background_tasks.add_task(_run_job, job_id, disease, body.max_drugs)
    return {"job_id": job_id, "disease": disease, "status": "queued"}


def _run_job(job_id: str, disease: str, max_drugs: int):
    JOBS[job_id]["status"] = "running"
    try:
        # Lazy import: rdkit, sklearn, xgboost, shap only loaded here
        from main import run_pipeline

        results_cache = os.path.join(OUTPUT_DIR, f"{disease}_results_cache.json")

        # Return cached result instantly if available
        if os.path.exists(results_cache):
            print(f"[cache] Loading results from disk for '{disease}'...")
            with open(results_cache) as f:
                raw = f.read()
            raw = _re.sub(r':\s*Infinity',  ': null', raw)
            raw = _re.sub(r':\s*-Infinity', ': null', raw)
            raw = _re.sub(r':\s*NaN',       ': null', raw)
            results = json.loads(raw)
            csv_ok  = os.path.exists(results.get("csv_path",  ""))
            xlsx_ok = os.path.exists(results.get("xlsx_path", ""))
            if csv_ok and xlsx_ok:
                JOBS[job_id]["status"]  = "complete"
                JOBS[job_id]["results"] = results
                print(f"[cache] Returned cached results instantly for '{disease}'.")
                return

        results = run_pipeline(disease, max_drugs)
        if "error" in results:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"]  = results["error"]
        else:
            JOBS[job_id]["status"]  = "complete"
            JOBS[job_id]["results"] = results

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"]  = str(e)
        print(f"[!] Job {job_id} failed: {e}")


# ==========================================
# STATUS
# ==========================================
@app.get("/status/{job_id}")
def status_endpoint(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id":  job_id,
        "disease": job["disease"],
        "status":  job["status"],
        "error":   job["error"],
    }


# ==========================================
# RESULTS — sanitise floats before JSON serialisation
# ==========================================
def _sanitize_for_json(obj):
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


@app.get("/results/{job_id}")
def results_endpoint(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "complete":
        raise HTTPException(status_code=202, detail=f"Job status: {job['status']}")
    return JSONResponse(content=_sanitize_for_json(job["results"]))


# ==========================================
# DOWNLOADS
# ==========================================
@app.get("/download/{job_id}")
def download_csv(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "complete":
        raise HTTPException(status_code=202, detail="Job not yet complete.")
    csv_path = job["results"].get("csv_path")
    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV file not found.")
    return FileResponse(
        path=csv_path,
        filename=f"{job['disease']}_results.csv",
        media_type="text/csv",
    )


@app.get("/download-excel/{job_id}")
def download_excel(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "complete":
        raise HTTPException(status_code=202, detail="Job not yet complete.")
    xlsx_path = job["results"].get("xlsx_path")
    if not xlsx_path or not os.path.exists(xlsx_path):
        raise HTTPException(status_code=404, detail="Excel file not found.")
    return FileResponse(
        path=xlsx_path,
        filename=f"{job['disease']}_results.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ==========================================
# LOCAL DEV
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("website:app", host=API_HOST, port=API_PORT, reload=True)
