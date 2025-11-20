# services/llm_service.py
import os
from pathlib import Path
from llama_cpp import Llama

MODEL_PATH = Path(__file__).parent.parent / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=1024,
    n_threads=os.cpu_count() or 4,
    verbose=False,
)

def chat_completion(messages: list):
    return llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.0,
    )

def summarize_job(description: str, skills: list[str]):
    if not description:
        return "No description available."

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
                or "AI summary unavailable."
            )

        except Exception as e:
            print(f"[summarize_job] Failed with limit={limit}: {e}")
            continue

    return "AI summary could not be generated due to context window limits."


