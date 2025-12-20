import traceback
import threading
import asyncio
from langchain_core.messages import AIMessageChunk

class AsyncWorker:
    def __init__(self):
        self._thread = None
        self._loop = None
        self._started = threading.Event()
        self._graph = None

    def start(self, init_graph_coroutine, init_timeout=300):
        if self._thread and self._thread.is_alive():
            return

        def _run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                try:
                    self._graph = loop.run_until_complete(init_graph_coroutine())
                except Exception as e:
                    print("Error initializing graph in worker:", e)
                    traceback.print_exc()
                    self._graph = None
                self._started.set()
                loop.run_forever()
            finally:
                pending = asyncio.all_tasks(loop=loop)
                for t in pending:
                    t.cancel()
                try:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
                loop.close()

        t = threading.Thread(target=_run_loop, daemon=True)
        t.start()
        self._thread = t
        if not self._started.wait(timeout=init_timeout):
            raise RuntimeError("AsyncWorker failed to initialize within timeout")

    def stop(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def graph(self):
        return self._graph

    def run_coro(self, coro):
        if not self._loop:
            raise RuntimeError("Worker loop not started")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def start_astream_task(self, messages, out_queue, config):
        if self._graph is None:
            raise RuntimeError("Graph not initialized in worker")

        async def _consume():
            try:
                async for event in self._graph.astream(messages, 
                                                       config=config, 
                                                       stream_mode="messages"):
                    # print(event)
                    try:
                        message, metadata = event
                        
                        if not isinstance(message, AIMessageChunk):
                            continue
                            
                        if metadata.get("langgraph_node") not in ["generate_query_or_respond", "generate_answer"]:
                            continue

                        text = message.content
                    # try:
                    #     text = event[0].content

                    except Exception as e:
                        text = str(event)
                        print(f"Error parsing event: {e}")
                    
                    if text:
                        out_queue.put(text)
                        
            except Exception as e:
                out_queue.put(f"[STREAM ERROR] {e}")
                traceback.print_exc()
            finally:
                out_queue.put(None)

        return self.run_coro(_consume())