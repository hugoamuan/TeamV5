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
        temperature=0.4,
    )

