import json
import logging
import asyncio
import queue as sync_queue
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from src.graph import init_graph
from src.utils.async_worker import AsyncWorker

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

worker = AsyncWorker()
try:
    worker.start(init_graph)
    logger.info("AsyncWorker started and graph initialized.")
except Exception as e:
    logger.exception("Failed to start AsyncWorker / initialize graph: %s", e)

class Query(BaseModel):
    query: str
    config: dict

class ResponseModel(BaseModel):
    answer: str

def sse_event(data: dict) -> str:
    """Return a Server-Sent Event (SSE) formatted string for given dict payload."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

async def text_generator(query_text: str, config: dict) -> AsyncGenerator[str, None]:
    if worker.graph is None:
        err = "Graph not initialized"
        logger.error(err)
        raise HTTPException(status_code=500, detail=err)

    q: sync_queue.Queue = sync_queue.Queue()
    try:
        future = worker.start_astream_task(
            {"messages": [{"role": "user", "content": query_text}]},
            config,
            q,
        )
    except Exception as e:
        logger.exception("Failed to schedule astream task on worker: %s", e)
        yield sse_event({"error": str(e)})
        return

    loop = asyncio.get_event_loop()

    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break

            if isinstance(item, str) and item.startswith("[STREAM ERROR]"):
                logger.error("Stream error from worker: %s", item)
                yield sse_event({"error": item})
                break

            yield sse_event({"context": item})
    except asyncio.CancelledError:
        logger.info("Client disconnected, cancelling stream.")
        try:
            future.cancel()
        except Exception:
            pass
        raise
    except Exception as e:
        logger.exception("Unexpected error in text_generator: %s", e)
        yield sse_event({"error": str(e)})
    finally:
        try:
            if future and not future.done():
                future.result(timeout=0.1)
        except Exception:
            pass

# --- FastAPI router ---
router = APIRouter()

@router.post("/chat/stream", response_model=None)
async def get_stream_answer(query: Query):
    """
    Stream chat response for incoming query as Server-Sent Events (text/event-stream).
    Each event is a JSON object: {"context": "..."} or {"error": "..."}.
    """
    if worker.graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")

    return StreamingResponse(
        text_generator(query.query, query.config),
        media_type="text/event-stream",
    )
