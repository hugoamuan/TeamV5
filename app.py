# app.py – Self-Hosted Llama + LinkedIn Job Scraper (Full & Short summaries)
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import os, time, hashlib, requests, logging

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')

# ---------- CONFIG ----------
MODEL_PATH = Path(__file__).parent / "models" / "Llama-3.2-1B-Instruct-f16.gguf"
CACHE_TTL = 60
JOB_CACHE = {}       # cache for search results
SUMMARY_CACHE = {}   # cache for short summaries
DETAIL_CACHE = {}    # cache for full descriptions

# ---------- LLM ----------
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=2048,
    n_threads=os.cpu_count() or 4,
    verbose=False,
)

# ---------- APP ----------
app = FastAPI(title="Self-Hosted AI + LinkedIn Job Scraper (RAW)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- HEALTH ----------
@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}

# ---------- MODEL INFO ----------
@app.get("/model-info")
def model_info():
    size = MODEL_PATH.stat().st_size
    sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    return {"model": str(MODEL_PATH), "size": size, "sha256": sha}

# ---------- CHAT COMPLETION ----------
@app.post("/v1/chat/completions")
def chat(req: dict):
    return llm.create_chat_completion(
        messages=req.get("messages", []),
        max_tokens=512,
        temperature=0.4,
    )

# ---------- LINKEDIN SCRAPER ----------
def fetch_linkedin_jobs(keyword: str, location: str, limit: int = 5):
    """Scrape LinkedIn search results (short summaries only)."""
    cache_key = f"linkedin|{keyword}|{location}|{limit}"
    cached = JOB_CACHE.get(cache_key)
    if cached and (time.time() - cached[0] < CACHE_TTL):
        logging.info(f"[fetch_linkedin_jobs] cache hit: {cache_key}")
        return cached[1]

    q, l = quote_plus(keyword.strip()), quote_plus(location.strip())
    url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={q}&location={l}&start=0"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-CA,en-US;q=0.9",
    }

    try:
        html = requests.get(url, headers=headers, timeout=15).text
    except Exception as e:
        logging.error(f"[fetch_linkedin_jobs] fetch error: {e}")
        return {"error": str(e), "jobs": []}

    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("li") or soup.select("div.base-card")
    jobs = []

    for card in cards:
        try:
            anchor = (
                card.select_one("a.base-card__full-link")
                or card.select_one("a.result-card__full-card-link")
                or card.select_one("a[href*='/jobs/view/']")
            )
            href = anchor.get("href") if anchor else None
            url_abs = href if (href and href.startswith("http")) else (f"https://www.linkedin.com{href}" if href else None)

            title_el = (
                card.select_one("h3")
                or card.select_one(".base-search-card__title")
                or card.select_one(".job-card-list__title")
            )
            title = title_el.get_text(" ", strip=True) if title_el else (anchor.get_text(" ", strip=True) if anchor else None)

            company_el = (
                card.select_one("h4")
                or card.select_one(".base-search-card__subtitle")
                or card.select_one(".job-card-container__company-name")
                or card.select_one(".job-card-list__company-name")
            )
            company = company_el.get_text(" ", strip=True) if company_el else None

            loc_el = (
                card.select_one(".job-search-card__location")
                or card.select_one(".base-search-card__metadata > .job-search-card__location")
                or card.select_one(".job-card-container__metadata-item")
            )
            loc = loc_el.get_text(" ", strip=True) if loc_el else None

            # --- short summary fetch (200 chars, cached) ---
            summary = None
            if url_abs:
                cached_sum = SUMMARY_CACHE.get(url_abs)
                if cached_sum and (time.time() - cached_sum[0] < CACHE_TTL):
                    summary = cached_sum[1]
                else:
                    try:
                        job_resp = requests.get(url_abs, headers=headers, timeout=10)
                        job_soup = BeautifulSoup(job_resp.text, "html.parser")
                        summary_el = (
                            job_soup.select_one(".show-more-less-html__markup")
                            or job_soup.select_one(".description__text")
                        )
                        if summary_el:
                            summary = summary_el.get_text(" ", strip=True)[:200]
                        SUMMARY_CACHE[url_abs] = (time.time(), summary)
                    except Exception as e:
                        logging.warning(f"[fetch_linkedin_jobs] failed summary fetch: {e}")

            if title and url_abs:
                jobs.append({
                    "title": title,
                    "company": company,
                    "location": loc,
                    "url": url_abs,
                    "summary": summary,
                })
            if len(jobs) >= limit:
                break
        except Exception as e:
            logging.warning(f"[fetch_linkedin_jobs] parse error: {e}")
            continue

    result = {"error": None, "jobs": jobs}
    JOB_CACHE[cache_key] = (time.time(), result)
    logging.info(f"[fetch_linkedin_jobs] returning {len(jobs)} jobs")
    return result

# ---------- JOB SEARCH ENDPOINT ----------
@app.get("/jobs/search")
def search_jobs(
    keyword: str = Query(..., description="e.g., software developer"),
    location: str = Query(..., description="e.g., Vancouver, BC"),
    limit: int = Query(5, ge=1, le=25),
):
    logging.info(f"[jobs/search] keyword={keyword} location={location} limit={limit}")
    return fetch_linkedin_jobs(keyword, location, limit)

# ---------- JOB DETAILS ENDPOINT ----------
@app.get("/jobs/details")
def job_details(url: str = Query(..., description="Full LinkedIn job URL")):
    """Return full job description for the given LinkedIn job URL."""
    cached = DETAIL_CACHE.get(url)
    if cached and (time.time() - cached[0] < CACHE_TTL):
        return cached[1]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-CA,en-US;q=0.9",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return {"error": f"LinkedIn HTTP {resp.status_code}"}
        soup = BeautifulSoup(resp.text, "html.parser")

        title = soup.select_one("h1") or soup.select_one(".topcard__title")
        company = soup.select_one(".topcard__org-name-link") or soup.select_one(".topcard__flavor")
        location = soup.select_one(".topcard__flavor--bullet") or soup.select_one(".sub-nav-cta__meta-text")

        desc_el = soup.select_one(".show-more-less-html__markup") or soup.select_one(".description__text")
        description = desc_el.get_text(" ", strip=True) if desc_el else None

        data = {
            "title": title.get_text(" ", strip=True) if title else None,
            "company": company.get_text(" ", strip=True) if company else None,
            "location": location.get_text(" ", strip=True) if location else None,
            "url": url,
            "description": description,
        }

        DETAIL_CACHE[url] = (time.time(), data)
        return data

    except Exception as e:
        logging.error(f"[job_details] fetch error: {e}")
        return {"error": str(e)}

