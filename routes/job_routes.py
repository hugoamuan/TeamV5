# routes/job_routes.py
# for job-related endpoints (Linkedin scraping + AI summary)
# All routes are versioned under /v1/jobs
# 
# Flow:
#  1. Scrape LinkedIn job posting
#  2. Fetch full job description
#  3. match job desc with user's skills
#  4. Summarize matching jobs with the hosted AI model


from fastapi import APIRouter
from pydantic import BaseModel
from services.scrape_service import fetch_linkedin_jobs
from services.match_service import job_matches_skills
from services.scrape_service import fetch_job_details
from services.llm_service import summarize_job

job_router = APIRouter(prefix="/v1/jobs")

# Request model for POST /v1/jobs/search_user
class UserSearch(BaseModel):
    job_wanted: str
    skills: list[str]
    location: str
    limit: int = 5

# Personalized job search using POST request
@job_router.post("/search_user")
def search_user(req: UserSearch):

    # fetch jobs using job_wanted as the keyword
    raw = fetch_linkedin_jobs(req.job_wanted, req.location, req.limit)


    if raw["error"]:
        return raw

    filtered = []

    # evaluate job listing
    for job in raw["jobs"]:
        details = fetch_job_details(job["url"])
        full_desc = details.get("description", "") or ""

        # check if job matches user skills, if so summarize the job
        if job_matches_skills(full_desc, req.skills):
            job["ai_summary"] = summarize_job(full_desc, req.skills) + f"\n\nJob URL: {job['url']}"
            filtered.append(job)

    return {
        "error": None,
        "jobs_found": len(raw["jobs"]),
        "jobs_filtered": len(filtered),
        "jobs": filtered
    }

# Raw job scraper endpoint for debugging and testing
@job_router.get("/search")
def search_jobs(
    keyword: str,
    location: str,
    limit: int = 5
):
    return fetch_linkedin_jobs(keyword, location, limit)

