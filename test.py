import asyncio
from src.utils.utils import new_uuid
from src.graph import init_graph

thread_id = new_uuid()
config = {"configurable": {"thread_id": thread_id}}

async def stream_response(graph, query: str, config: dict):
    agen = graph.astream(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
        stream_mode="messages"
    )

    try:
        async for event in agen:
            # debug: print full event if cần
            # print("EVENT:", repr(event))
            text = event[0].content
            print(text)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            await agen.aclose()
        except Exception:
            pass

async def main():
    graph = await init_graph()
    await stream_response(graph, "Hồ Quý Ly sinh năm bao nhiêu?", config)

if __name__ == "__main__":
    asyncio.run(main())
