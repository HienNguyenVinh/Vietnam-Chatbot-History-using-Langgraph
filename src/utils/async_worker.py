import traceback
import threading
import asyncio

class AsyncWorker:
    def __init__(self):
        self._thread = None
        self._loop = None
        self._started = threading.Event()
        self._graph = None

    def start(self, init_graph_coroutine, init_timeout=300):
        """Start worker thread and initialize graph via init_graph_coroutine (async callable)."""
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
        """Schedule coroutine on worker loop and return concurrent.futures.Future"""
        if not self._loop:
            raise RuntimeError("Worker loop not started")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def start_astream_task(self, input_obj, config, out_queue):
        """Start a coroutine in worker loop to consume graph.astream and push chunks to out_queue."""
        if self._graph is None:
            raise RuntimeError("Graph not initialized in worker")

        async def _consume():
            try:
                async for event in self._graph.astream(input_obj, config=config, stream_mode="messages"):
                    try:
                        text = event[0].content
                    except Exception:
                        text = str(event)
                    out_queue.put(text)
            except Exception as e:
                out_queue.put(f"[STREAM ERROR] {e}")
                traceback.print_exc()
            finally:
                out_queue.put(None)

        return self.run_coro(_consume())
    
    async def _delete_thread_coro(self, thread_id: str) -> bool:
        """
        Coroutine that runs on worker loop to delete all checkpoints for given thread_id.
        Returns True on success, False on failure.
        """
        if self._graph is None:
            return False
        cp = getattr(self._graph, "checkpointer", None)
        if cp is None:
            return False

        try:
            if hasattr(cp, "adelete_thread") and asyncio.iscoroutinefunction(getattr(cp, "adelete_thread")):
                await cp.adelete_thread(thread_id)
                return True
        except Exception:
            traceback.print_exc()

        try:
            if hasattr(cp, "delete_thread"):
                maybe = getattr(cp, "delete_thread")
                if asyncio.iscoroutinefunction(maybe):
                    await maybe(thread_id)
                else:
                    maybe(thread_id)
                return True
        except Exception:
            traceback.print_exc()

        try:
            checkpoints = None
            if hasattr(cp, "alist") and asyncio.iscoroutinefunction(getattr(cp, "alist")):
                checkpoints = await cp.alist(None)
            elif hasattr(cp, "list"):
                checkpoints = cp.list(None)
            else:
                checkpoints = []

            to_delete = []
            for ck in checkpoints or []:
                cfg = getattr(ck, "config", None) or (ck.get("config") if isinstance(ck, dict) else {})
                if cfg and cfg.get("configurable", {}).get("thread_id") == thread_id:
                    # try to get id
                    ck_id = getattr(ck, "id", None) or (ck.get("id") if isinstance(ck, dict) else None)
                    if ck_id:
                        to_delete.append(ck_id)

            for ckid in to_delete:
                if hasattr(cp, "adelete") and asyncio.iscoroutinefunction(getattr(cp, "adelete")):
                    await cp.adelete(ckid)
                elif hasattr(cp, "delete"):
                    maybe = getattr(cp, "delete")
                    if asyncio.iscoroutinefunction(maybe):
                        await maybe(ckid)
                    else:
                        maybe(ckid)
            return True
        except Exception:
            traceback.print_exc()
            return False

    def delete_thread(self, thread_id: str):
        """
        Schedule deletion coroutine on worker loop.
        Returns concurrent.futures.Future which yields True/False.
        """
        if not self._loop:
            raise RuntimeError("Worker loop not started")
        return asyncio.run_coroutine_threadsafe(self._delete_thread_coro(thread_id), self._loop)