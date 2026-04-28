# ==========================================
# website.py — FastAPI backend
#
# Install: pip install fastapi uvicorn python-multipart
# Run:     uvicorn website:app --reload --host 0.0.0.0 --port 8000
#
# Endpoints:
#   POST /validate          — validate disease input
#   POST /run               — run full pipeline (async job)
#   GET  /status/{job_id}   — poll job status
#   GET  /results/{job_id}  — get completed results
#   GET  /download/{job_id} — download CSV
#   GET  /health            — health check
# ==========================================

import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

from config import API_HOST, API_PORT, CORS_ORIGINS, MAX_DRUGS
from validator import validate_disease_input
from main import run_pipeline

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

# In-memory job store (use Redis in production)
JOBS: dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=4)


# ==========================================
# SCHEMAS
# ==========================================
class ValidateRequest(BaseModel):
    disease: str

class RunRequest(BaseModel):
    disease: str
    max_drugs: Optional[int] = MAX_DRUGS
    force_proceed: Optional[bool] = False


# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serves the frontend index.html — works both locally and on Railway."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/validate")
def validate_endpoint(body: ValidateRequest):
    """
    Layer 1-3 validation. Call this before /run.
    Frontend uses the response to show suggestions or errors
    before triggering the expensive pipeline.

    Response shape:
      {
        valid: bool,
        disease: str | null,
        drug_count: int,
        suggestions: list[str],
        warnings: list[str],
        errors: list[str],
      }
    """
    result = validate_disease_input(body.disease)
    return JSONResponse(content=result)


@app.post("/run")
def run_endpoint(body: RunRequest, background_tasks: BackgroundTasks):
    """
    Kicks off the full pipeline as a background job.
    Returns a job_id immediately — poll /status/{job_id}.

    The pipeline can take 2-10 minutes for 100 drugs
    due to PubChem and pkCSM API calls.
    """
    # Validate first
    validation = validate_disease_input(body.disease, force_proceed=body.force_proceed)
    if not validation["valid"]:
        raise HTTPException(status_code=422, detail={
            "errors": validation["errors"],
            "suggestions": validation["suggestions"],
        })

    disease = validation["disease"]
    job_id = str(uuid.uuid4())[:8]

    JOBS[job_id] = {
        "id":       job_id,
        "disease":  disease,
        "status":   "queued",
        "progress": 0,
        "results":  None,
        "error":    None,
    }

    background_tasks.add_task(_run_job, job_id, disease, body.max_drugs)
    return {"job_id": job_id, "disease": disease, "status": "queued"}


def _run_job(job_id: str, disease: str, max_drugs: int):
    """Runs in background thread. Updates JOBS dict throughout."""
    JOBS[job_id]["status"] = "running"
    try:
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


@app.get("/status/{job_id}")
def status_endpoint(job_id: str):
    """
    Poll this endpoint to track job progress.
    Frontend should poll every 3-5 seconds.

    status values: queued | running | complete | failed
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id":  job_id,
        "disease": job["disease"],
        "status":  job["status"],
        "error":   job["error"],
    }


@app.get("/results/{job_id}")
def results_endpoint(job_id: str):
    """
    Returns full results once status is 'complete'.
    Includes correlation matrix, ML performance, SHAP values,
    drug filter stats, and a 10-row data preview.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "complete":
        raise HTTPException(status_code=202, detail=f"Job status: {job['status']}")
    return job["results"]


@app.get("/download/{job_id}")
def download_endpoint(job_id: str):
    """
    Returns the full results CSV for download.
    """
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


# ==========================================
# RUN DIRECTLY
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("website:app", host=API_HOST, port=API_PORT, reload=True)
