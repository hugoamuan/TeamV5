# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time, hashlib
from pathlib import Path
from routes.chat_routes import chat_router
from routes.job_routes import job_router

MODEL_PATH = Path(__file__).parent / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

app = FastAPI(title="Self-Hosted AI + LinkedIn Job Scraper (RAW)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(job_router)

@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}

@app.get("/model-info")
def model_info():
    size = MODEL_PATH.stat().st_size
    sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    return {"model": str(MODEL_PATH), "size": size, "sha256": sha}


