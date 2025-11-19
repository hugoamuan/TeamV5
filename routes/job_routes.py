# routes/job_routes.py

from fastapi import APIRouter
from pydantic import BaseModel
from services.scrape_service import fetch_linkedin_jobs
from services.match_service import job_matches_skills
from services.scrape_service import fetch_job_details

job_router = APIRouter(prefix="/jobs")

class UserSearch(BaseModel):
    job_wanted: str
    skills: list[str]
    location: str
    limit: int = 5

@job_router.post("/search_user")
def search_user(req: UserSearch):
    # fetch jobs using job_wanted as the keyword
    raw = fetch_linkedin_jobs(req.job_wanted, req.location, req.limit)

    if raw["error"]:
        return raw

    filtered = []
    for job in raw["jobs"]:
        details = fetch_job_details(job["url"])
        full_desc = details.get("description", "") or ""

        if job_matches_skills(full_desc, req.skills):
            filtered.append(job)

    return {
        "error": None,
        "jobs_found": len(raw["jobs"]),
        "jobs_filtered": len(filtered),
        "jobs": filtered
    }

@job_router.get("/search")
def search_jobs(
    keyword: str,
    location: str,
    limit: int = 5
):
    return fetch_linkedin_jobs(keyword, location, limit)

