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

# [{'role': 'system', 'content': '\n'}, 
#  HumanMessage(content='Chiến dịch Điện biên phủ diễn ra trong bao nhiêu ngày đêm', additional_kwargs={}, response_metadata={}, id='34731b8e-11ee-43d4-a216-f46cf6bc0e71'), 
#  HumanMessage(content='Bác hồ ra đi tìm đường cứu nước trong bao nhiêu năm', additional_kwargs={}, response_metadata={}, id='15619bcc-fe9b-4358-bd7c-caeef8521f3f'), 
#  HumanMessage(content='có', additional_kwargs={}, response_metadata={}, id='a4510b76-cce9-48cc-8805-6646fe135cd3')]