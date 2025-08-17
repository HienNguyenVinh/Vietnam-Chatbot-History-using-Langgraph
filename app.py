import asyncio
from src.utils.utils import new_uuid
from src.graph import graph

thread_id = new_uuid()
config = {"configurable": {"thread_id": thread_id}}

async def stream_response(query: str, config: dict):
    async for event in graph.astream(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
        stream_mode="messages"
    ):
        text = event[0].content
        print(text)

if __name__ == "__main__":
    asyncio.run(stream_response("Hồ Quý Ly sinh năm bao nhiêu?", config))