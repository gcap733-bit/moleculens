# MolecuLens — Deployment Guide (Railway)
# ==========================================
# Result: a permanent shareable link like
# https://moleculens-production.up.railway.app
# Free tier, no credit card needed, ~5 minutes setup.
# ==========================================


# ==========================================
# STEP 1 — Add these 3 files to your project folder
# ==========================================

# Your folder should look like this:
# drug_pipeline/
#   config.py
#   validator.py
#   fetcher.py
#   ml.py
#   main.py
#   website.py
#   index.html
#   requirements.txt   ← already have this
#   Procfile           ← CREATE THIS (see below)
#   railway.toml       ← CREATE THIS (see below)
#   .gitignore         ← CREATE THIS (see below)


# ──────────────────────────────────────────
# FILE: Procfile  (no extension, exactly this name)
# ──────────────────────────────────────────
# Contents:
#
#   web: uvicorn website:app --host 0.0.0.0 --port $PORT
#
# (Railway sets $PORT automatically)


# ──────────────────────────────────────────
# FILE: railway.toml
# ──────────────────────────────────────────
# Contents:
#
#   [build]
#   builder = "nixpacks"
#
#   [deploy]
#   startCommand = "uvicorn website:app --host 0.0.0.0 --port $PORT"
#   restartPolicyType = "on_failure"


# ──────────────────────────────────────────
# FILE: .gitignore
# ──────────────────────────────────────────
# Contents:
#
#   .cache/
#   outputs/
#   __pycache__/
#   *.pyc
#   .env


# ==========================================
# STEP 2 — Serve index.html from FastAPI
# ==========================================
# Railway only exposes ONE port. So FastAPI must
# serve the frontend HTML too.
#
# Add this to the TOP of website.py, after the imports:
#
#   from fastapi.responses import HTMLResponse
#   from fastapi.staticfiles import StaticFiles
#   import os
#
# Then add this route BEFORE the other endpoints:
#
#   @app.get("/", response_class=HTMLResponse)
#   def serve_frontend():
#       with open("index.html", "r") as f:
#           return f.read()
#
# And update index.html — change the API line from:
#   const API = 'http://localhost:8000';
# to:
#   const API = '';   // empty = same server, works on any domain


# ==========================================
# STEP 3 — Push to GitHub
# ==========================================
# Railway deploys from GitHub. Do this once:
#
# 1. Go to https://github.com and create a free account if needed
# 2. Click "New repository" → name it "moleculens" → Public → Create
# 3. Open Command Prompt in your drug_pipeline folder:

#    cd path\to\drug_pipeline

#    git init
#    git add .
#    git commit -m "initial deploy"
#    git branch -M main
#    git remote add origin https://github.com/YOUR_USERNAME/moleculens.git
#    git push -u origin main

# Replace YOUR_USERNAME with your actual GitHub username.
# GitHub will ask you to log in the first time.


# ==========================================
# STEP 4 — Deploy on Railway
# ==========================================
# 1. Go to https://railway.app
# 2. Click "Start a New Project"
# 3. Choose "Deploy from GitHub repo"
# 4. Connect your GitHub account → select "moleculens"
# 5. Railway auto-detects Python and starts building
# 6. Wait ~3 minutes for build to finish
# 7. Click "Settings" → "Networking" → "Generate Domain"
# 8. You get a link like: https://moleculens-production.up.railway.app


# ==========================================
# STEP 5 — Share the link
# ==========================================
# Send that link to anyone. They open it in their browser,
# type a disease name, and the pipeline runs on Railway's
# servers. Your PC does not need to be on.
#
# Free tier gives you 500 hours/month — plenty for sharing
# with a research group or supervisor.


# ==========================================
# UPDATING THE SITE LATER
# ==========================================
# Every time you change code, just run:
#
#   git add .
#   git commit -m "describe your change"
#   git push
#
# Railway auto-redeploys within ~2 minutes.


# ==========================================
# IMPORTANT: Cache on Railway
# ==========================================
# Railway's filesystem resets on each deploy, so the
# .cache/ folder won't persist between restarts.
# This means API calls (PubChem, pkCSM) re-run fresh
# each time someone searches.
#
# For now this is fine. If you want persistent caching
# later, add a Railway Redis plugin (free tier available)
# and we can swap the cache backend — just ask.


# ==========================================
# TROUBLESHOOTING
# ==========================================
# Build fails?
#   → Check the build logs in Railway dashboard
#   → Most common cause: missing package in requirements.txt
#
# Site loads but pipeline errors?
#   → Click "View Logs" in Railway dashboard
#   → Look for the error message from your Python code
#
# "Application failed to respond"?
#   → Make sure Procfile has exactly:
#      web: uvicorn website:app --host 0.0.0.0 --port $PORT
#   → Make sure website.py has the "/" route serving index.html
