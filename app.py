# app.py – Self-Hosted Llama + LinkedIn Job Scraper (Full & Short summaries)
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

# Internal: Scrape LinkedIn for job cards + short summaries
def fetch_linkedin_jobs(keyword: str, location: str, limit: int = 5):
    """
    Scrapes LinkedIn job listings based on search criteria.
    
    Args:
        keyword: Job title or keywords to search for
        location: Geographic location for job search
        limit: Maximum number of jobs to return (default 5)
    
    Returns:
        Dictionary with 'error' and 'jobs' list containing job postings
    """
    # Create unique cache key based on search parameters
    cache_key = f"linkedin|{keyword}|{location}|{limit}"
    
    # Check if we have recent cached results (within CACHE_TTL seconds)
    cached = JOB_CACHE.get(cache_key)
    if cached and (time.time() - cached[0] < CACHE_TTL):
        logging.info(f"[fetch_linkedin_jobs] cache hit: {cache_key}")
        return cached[1]  # Return cached data

    # URL-encode search parameters for safe inclusion in URL
    q, l = quote_plus(keyword.strip()), quote_plus(location.strip())
    
    # Build LinkedIn job search URL with encoded parameters
    url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={q}&location={l}&start=0"
    
    # Set headers to mimic a real browser request (helps avoid blocking)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-CA,en-US;q=0.9",
    }

    try:
        # Fetch the LinkedIn search results page
        html = requests.get(url, headers=headers, timeout=15).text
    except Exception as e:
        # If request fails, log error and return empty result
        logging.error(f"[fetch_linkedin_jobs] fetch error: {e}")
        return {"error": str(e), "jobs": []}

    # Parse HTML content with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    
    # Try multiple selectors to find job cards (LinkedIn HTML structure varies)
    cards = soup.select("li") or soup.select("div.base-card")
    jobs = []

    # Process each job card
    for card in cards:
        try:
            # Find the main job link using multiple possible selectors
            anchor = (
                card.select_one("a.base-card__full-link")
                or card.select_one("a.result-card__full-card-link")
                or card.select_one("a[href*='/jobs/view/']")
            )
            
            # Extract href and ensure it's an absolute URL
            href = anchor.get("href") if anchor else None
            url_abs = href if (href and href.startswith("http")) else (f"https://www.linkedin.com{href}" if href else None)

            # Extract job title from various possible elements
            title_el = (
                card.select_one("h3")
                or card.select_one(".base-search-card__title")
                or card.select_one(".job-card-list__title")
            )
            title = title_el.get_text(" ", strip=True) if title_el else (anchor.get_text(" ", strip=True) if anchor else None)

            # Extract company name from various possible elements
            company_el = (
                card.select_one("h4")
                or card.select_one(".base-search-card__subtitle")
                or card.select_one(".job-card-container__company-name")
                or card.select_one(".job-card-list__company-name")
            )
            company = company_el.get_text(" ", strip=True) if company_el else None

            # Extract location from various possible elements
            loc_el = (
                card.select_one(".job-search-card__location")
                or card.select_one(".base-search-card__metadata > .job-search-card__location")
                or card.select_one(".job-card-container__metadata-item")
            )
            loc = loc_el.get_text(" ", strip=True) if loc_el else None

            # Fetch or retrieve cached short job description
            summary = None
            if url_abs:
                # Check if we have a cached summary for this job URL
                cached_sum = SUMMARY_CACHE.get(url_abs)
                if cached_sum and (time.time() - cached_sum[0] < CACHE_TTL):
                    summary = cached_sum[1]  # Use cached summary
                else:
                    try:
                        # Fetch the full job page to get description
                        job_resp = requests.get(url_abs, headers=headers, timeout=10)
                        job_soup = BeautifulSoup(job_resp.text, "html.parser")
                        
                        # Find description element
                        summary_el = (
                            job_soup.select_one(".show-more-less-html__markup")
                            or job_soup.select_one(".description__text")
                        )
                        
                        # Extract first 200 characters as summary
                        if summary_el:
                            summary = summary_el.get_text(" ", strip=True)[:200]
                        
                        # Cache the summary for future use
                        SUMMARY_CACHE[url_abs] = (time.time(), summary)
                    except Exception as e:
                        logging.warning(f"[fetch_linkedin_jobs] failed summary fetch: {e}")

            # Only add job if we have at least title and URL
            if title and url_abs:
                jobs.append({
                    "title": title,
                    "company": company,
                    "location": loc,
                    "url": url_abs,
                    "summary": summary,
                })
            
            # Stop if we've reached the requested limit
            if len(jobs) >= limit:
                break
        except Exception as e:
            # Log parsing errors but continue processing other jobs
            logging.warning(f"[fetch_linkedin_jobs] parse error: {e}")
            continue

    # Build result object
    result = {"error": None, "jobs": jobs}
    
    # Cache the results with current timestamp
    JOB_CACHE[cache_key] = (time.time(), result)
    logging.info(f"[fetch_linkedin_jobs] returning {len(jobs)} jobs")
    
    return result

# Public endpoint for job search
@app.get("/jobs/search")
def search_jobs(
    keyword: str = Query(..., description="e.g., software developer"),
    location: str = Query(..., description="e.g., Vancouver, BC"),
    limit: int = Query(5, ge=1, le=25),
):
    """
    Public API endpoint for searching LinkedIn jobs.
    
    Query Parameters:
        keyword: Job search keywords (required)
        location: Geographic location (required)
        limit: Number of results (1-25, default 5)
    """
    logging.info(f"[jobs/search] keyword={keyword} location={location} limit={limit}")
    
    # Delegate to internal scraping function
    return fetch_linkedin_jobs(keyword, location, limit)

# Public endpoint for full job description details
@app.get("/jobs/details")
def job_details(url: str = Query(..., description="Full LinkedIn job URL")):
    """
    Fetches complete details for a specific job posting.
    
    Query Parameters:
        url: Full LinkedIn job URL (required)
    
    Returns:
        Dictionary with job title, company, location, and full description
    """
    # Check if we have cached details for this URL
    cached = DETAIL_CACHE.get(url)
    if cached and (time.time() - cached[0] < CACHE_TTL):
        return cached[1]  # Return cached details

    # Set browser-like headers
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-CA,en-US;q=0.9",
    }

    try:
        # Fetch the job posting page
        resp = requests.get(url, headers=headers, timeout=15)
        
        # Check if request was successful
        if resp.status_code != 200:
            return {"error": f"LinkedIn HTTP {resp.status_code}"}
        
        # Parse the HTML response
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract job title from page
        title = soup.select_one("h1") or soup.select_one(".topcard__title")
        
        # Extract company name
        company = soup.select_one(".topcard__org-name-link") or soup.select_one(".topcard__flavor")
        
        # Extract location
        location = soup.select_one(".topcard__flavor--bullet") or soup.select_one(".sub-nav-cta__meta-text")

        # Extract full job description
        desc_el = soup.select_one(".show-more-less-html__markup") or soup.select_one(".description__text")
        description = desc_el.get_text(" ", strip=True) if desc_el else None

        # Build response object with extracted data
        data = {
            "title": title.get_text(" ", strip=True) if title else None,
            "company": company.get_text(" ", strip=True) if company else None,
            "location": location.get_text(" ", strip=True) if location else None,
            "url": url,
            "description": description,
        }

        # Cache the details for future requests
        DETAIL_CACHE[url] = (time.time(), data)
        
        return data

    except Exception as e:
        # Log and return error if fetching fails
        logging.error(f"[job_details] fetch error: {e}")
        return {"error": str(e)}
