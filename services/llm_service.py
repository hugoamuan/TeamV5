# services/llm_service.py
# local LLM service wrapper for AI operations
# This module loads the local GGUF model and exposes chat_completion() and summarize_job()
# - helps organize FastAPI endpoints in one dedicated place and lightweight
import os
from pathlib import Path
from llama_cpp import Llama
from .strings import STRINGS

MODEL_PATH = Path(__file__).parent.parent / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# Initialize the model once at import time since it is expensive to do per request
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=1024,
    n_threads=os.cpu_count() or 4,
    verbose=False,
)

# OpenAI-style chat format 
def chat_completion(messages: list):
    return llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.0,
    )

# Summarizes a job posting into one paragraph using the local LLM
# Handles:
#  - Chunking long descriptions to avoid exceeding context size
#  - avoid hallucinations (plausible but false or fabricated answers)
def summarize_job(description: str, skills: list[str]):
    if not description:
        return STRINGS.NO_DESCRIPTION

    # limit the desc size to avoid timeouts
    description = description[:3000]

    skill_list = ", ".join(skills)

    # progressively smaller text chunks
    limits = [2500, 1800, 1200, 800, 500]

    for limit in limits:
        try:
            desc_chunk = description[:limit]

            prompt = (
               "Summarize the following job posting one paragraph.\n"
               "RULES:\n"
               "- DO NOT make up or guess any technologies.\n"
               "- ONLY use information that appears in the text.\n"
               "- If a section is missing, write 'Not specified'.\n"
               "- Relate the users skills with the job posting if possible.\n"
               "User skills: {skills}\n\n"
              "Job Description:\n{desc}"
              ).format(
                  skills=", ".join(skills),
                  desc=description
              )


            resp = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )

            choice = resp["choices"][0]
            return (
                choice.get("message", {}).get("content")
                or choice.get("text")
                or STRINGS.SUMMMARY_FAIL
            )

        except Exception as e:
            print(f"[summarize_job] Failed with limit={limit}: {e}")
            continue

    return STRINGS.SUMMARY_FAIL


