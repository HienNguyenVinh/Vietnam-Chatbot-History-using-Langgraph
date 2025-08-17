import asyncio
from src.utils.utils import new_uuid
from src.graph import graph, init_checkpointer

thread_id = new_uuid()
config = {"configurable": {"thread_id": thread_id}}

async def stream_response(query: str, config: dict):
    graph.checkpointer = await init_checkpointer("chatbot.db")
    try:
        async for event in graph.astream(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
            stream_mode="messages"
        ):
            text = event[0].content
            print(text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(stream_response("Hồ Quý Ly sinh năm bao nhiêu?", config))