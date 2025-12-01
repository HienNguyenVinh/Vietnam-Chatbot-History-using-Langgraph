import json
import logging
import asyncio
import queue as sync_queue
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Body, Query as FastAPIQuery
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from src.graph import init_graph
from src.utils.async_worker import AsyncWorker
from src.utils.utils import new_uuid
from src.db_helper.chat_history_service import get_chat_history, clear_chat_history, format_chat_history, save_message, get_threads_by_user


load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

worker = AsyncWorker()
try:
    worker.start(init_graph)
    logger.info("AsyncWorker started and graph initialized.")
except Exception as e:
    logger.exception("Failed to start AsyncWorker / initialize graph: %s", e)


class ChatQuery(BaseModel):
    query: str
    thread_id: str
    user_id: int
    config: Optional[dict] = {}

class ThreadCreate(BaseModel):
    user_id: int


def sse_event(data: dict) -> str:
    """Return a Server-Sent Event (SSE) formatted string for given dict payload."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def text_generator(query_text: str, thread_id: str, user_id: str) -> AsyncGenerator[str, None]:
    if worker.graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")

    raw_history = await get_chat_history(thread_id, user_id)
    formatted_history = format_chat_history(raw_history)
    messages = formatted_history + [{"role": "user", "content": query_text}]

    q: sync_queue.Queue = sync_queue.Queue()

    try:
        future = worker.start_astream_task(
            {"messages": messages},
            q
        )
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        return

    loop = asyncio.get_running_loop()
    full_resposne = ""

    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break

            if isinstance(item, str) and item.startswith("[STREAM ERROR]"):
                yield f"data: {json.dumps({'error': item}, ensure_ascii=False)}\n\n"
                break
            full_resposne += item
            
            if item:
                yield f"data: {json.dumps({'context': item}, ensure_ascii=False)}\n\n"

        if full_resposne:
            await save_message(thread_id, user_id, query_text, full_resposne)

    except asyncio.CancelledError:
        try:
            future.cancel()
        except Exception:
            pass
        raise
    except Exception as e:
        logger.exception("Error during streaming")
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"


# --- router ---
router = APIRouter()

@router.get("/threads")
async def list_threads(user_id: int=FastAPIQuery(..., description="ID người dùng")):
    threads = await get_threads_by_user(user_id)
    return {"threads": threads}

@router.post("/threads")
async def create_thread(request: ThreadCreate):
    new_thread_id = new_uuid()
    return {
        "thread_id": new_thread_id,
        "user_id": request.user_id,
        "message": "Thread created successfully!"
    }

@router.get("/threads/{thread_id}")
async def get_thread_conversation(thread_id: str, user_id: str=FastAPIQuery(...)):
    chat_history = await get_chat_history(thread_id, user_id)
    formatted = format_chat_history(chat_history)
    return {
        "thread_id": thread_id,
        "messages": formatted
    }

@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str, user_id: int=FastAPIQuery(...)):
    result = await clear_chat_history(thread_id, user_id)
    if result["type"] == "failed":
        raise HTTPException(status_code=500, detail="Failed to delete thread")
    return {"status": "success", "message": f"Thread {thread_id} deleted"}

@router.post("/chat/stream", response_model=None)
async def get_stream_answer(query: ChatQuery):
    if worker.graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")

    return StreamingResponse(
        text_generator(query.query, query.thread_id, query.user_id),
        media_type="text/event-stream",
    )
