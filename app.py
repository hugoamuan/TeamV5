from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import os, time, hashlib, requests, logging

# Enable basic request + cache event logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')

# Path and runtime config for local LLM + caching
MODEL_PATH = Path(__file__).parent / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
CACHE_TTL = 60  # cache lifetime in seconds
JOB_CACHE = {}       # Linkedin search results cache
SUMMARY_CACHE = {}   # short description summary cache
DETAIL_CACHE = {}    # full job description cache

# Load llama.cpp model locally (no OpenAI API)
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=1024,                     # max context window
    n_threads=os.cpu_count() or 4,  # auto-optimize concurrency
    verbose=False,
)

# FastAPI server entrypoint
app = FastAPI(title="Self-Hosted AI + LinkedIn Job Scraper (RAW)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # allow frontend from anywhere (dev use)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint for deployment probes
@app.get("/health")
def health():
    """
    Simple health check endpoint that returns current status and timestamp.
    Used by deployment systems to verify the service is running.
    """
    return {"status": "ok", "time": time.time()}

# Return info about currently loaded local model
@app.get("/model-info")
def model_info():
    """
    Returns metadata about the loaded LLM model including:
    - File path to the model
    - File size in bytes
    - SHA-256 hash for verification
    """
    # Get file size in bytes
    size = MODEL_PATH.stat().st_size

    # Calculate SHA-256 hash by reading entire model file
    sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    return {"model": str(MODEL_PATH), "size": size, "sha256": sha}

# Chat completions endpoint matching OpenAI format
@app.post("/v1/chat/completions")
def chat(req: dict):
    """
    OpenAI-compatible chat endpoint that uses the local Llama model.
    Accepts a request with 'messages' array and returns AI-generated response.
    """
    # Extract messages from request and pass to local LLM
    # Returns completion with controlled randomness (temperature=0.4)
    return llm.create_chat_completion(
        messages=req.get("messages", []),
        max_tokens=256,
        temperature=0.4,
    )

