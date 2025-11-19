# routes/chat_routes.py
from fastapi import APIRouter
from services.llm_service import chat_completion

chat_router = APIRouter(prefix="/v1")

@chat_router.post("/chat/completions")
def chat(req: dict):
    messages = req.get("messages", [])
    return chat_completion(messages)

