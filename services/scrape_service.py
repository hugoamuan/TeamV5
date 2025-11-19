# services/scrape_service.py

import time
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

# Cache time-to-live (in seconds)
CACHE_TTL = 60

# In-memory caches
JOB_CACHE = {}
SUMMARY_CACHE = {}
DETAIL_CACHE = {}

# ------------------------------
# Helper: resolve job description elements
# ------------------------------

def extract_description_element(soup):
    """
    LinkedIn frequently changes class names. This function tries
    multiple possible selectors to extract job descriptions.
    """

    return (
        soup.select_one(".show-more-less-html__markup") or
        soup.select_one(".description__text") or
        soup.select_one("div[data-test-job-description-text]") or
        soup.select_one(".job-details") or
        soup.select_one("#job-details") or
        soup.select_one(".decorated-job-posting__details") or
        soup.select_one(".core-section-container") or
        soup.find("section", {"class": lambda x: x and "description" in x})
    )


# ------------------------------
# Fetch LinkedIn Job Cards
# ------------------------------

def fetch_linkedin_jobs(keyword: str, location: str, limit: int = 5):
    """
    Scrapes LinkedIn job listings based on search criteria.
    """

    cache_key = f"linkedin|{keyword}|{location}|{limit}"

    # Use cached if fresh
    cached = JOB_CACHE.get(cache_key)
    if cached and (time.time() - cached[0] < CACHE_TTL):
        logging.info(f"[fetch_linkedin_jobs] cache hit: {cache_key}")
        return cached[1]

    # Encode search params
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
            # Extract job link
            anchor = (
                card.select_one("a.base-card__full-link")
                or card.select_one("a.result-card__full-card-link")
                or card.select_one("a[href*='/jobs/view/']")
            )

            href = anchor.get("href") if anchor else None
            url_abs = (
                href if (href and href.startswith("http"))
                else (f"https://www.linkedin.com{href}" if href else None)
            )

            # Job title
            title_el = (
                card.select_one("h3")
                or card.select_one(".base-search-card__title")
                or card.select_one(".job-card-list__title")
            )
            title = title_el.get_text(" ", strip=True) if title_el else (
                anchor.get_text(" ", strip=True) if anchor else None
            )

            # Company
            company_el = (
                card.select_one("h4")
                or card.select_one(".base-search-card__subtitle")
                or card.select_one(".job-card-container__company-name")
                or card.select_one(".job-card-list__company-name")
            )
            company = company_el.get_text(" ", strip=True) if company_el else None

            # Location
            loc_el = (
                card.select_one(".job-search-card__location")
                or card.select_one(".base-search-card__metadata > .job-search-card__location")
                or card.select_one(".job-card-container__metadata-item")
            )
            loc = loc_el.get_text(" ", strip=True) if loc_el else None

            # Summary (short job description)
            summary = None
            if url_abs:
                cached_sum = SUMMARY_CACHE.get(url_abs)
                if cached_sum and (time.time() - cached_sum[0] < CACHE_TTL):
                    summary = cached_sum[1]
                else:
                    try:
                        job_resp = requests.get(url_abs, headers=headers, timeout=10)
                        job_soup = BeautifulSoup(job_resp.text, "html.parser")

                        summary_el = extract_description_element(job_soup)

                        if summary_el:
                            summary = summary_el.get_text(" ", strip=True)

                        SUMMARY_CACHE[url_abs] = (time.time(), summary)
                    except Exception as e:
                        logging.warning(f"[fetch_linkedin_jobs] failed summary fetch: {e}")

            # Only add if title + URL exist
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

    # Cache results
    JOB_CACHE[cache_key] = (time.time(), result)
    logging.info(f"[fetch_linkedin_jobs] returning {len(jobs)} jobs")

    return result


# ------------------------------
# Fetch Full Job Details
# ------------------------------

def fetch_job_details(url: str):
    """
    Fetches full job description for a given job URL.
    Includes caching.
    """

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

        desc_el = extract_description_element(soup)
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

