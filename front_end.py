import streamlit as st
import threading
import traceback
import queue
import asyncio
import atexit
from concurrent.futures import TimeoutError as FutureTimeout
from langchain_core.messages import HumanMessage
from src.utils.utils import new_uuid
from src.graph import init_graph

# --------------------------- Helpers / Utilities ---------------------------

def generate_thread_id():
    return new_uuid()

def add_thread(thread_id):
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = []
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def load_conversation_sync_from_worker(worker, thread_id, timeout=5):
    """Load conversation from graph state running inside worker.
       Uses worker.run_coro to run graph.get_state(...) in the worker loop."""
    try:
        graph_obj = worker.graph
        if graph_obj is None:
            return []
        if hasattr(graph_obj, "get_state"):
            try:
                state = graph_obj.get_state(config={'configurable': {'thread_id': thread_id}})
                return state.values.get('messages', [])
            except Exception:
                pass
        if hasattr(graph_obj, "get_state"):
            fut = worker.run_coro(graph_obj.get_state({'configurable': {'thread_id': thread_id}}))
            return fut.result(timeout=timeout).values.get('messages', [])
    except FutureTimeout:
        return []
    except Exception:
        traceback.print_exc()
        return []

def retrieve_all_threads_from_worker(worker, timeout=5):
    """Return list of thread_ids from worker's checkpointer (safe sync wrapper)."""
    try:
        graph_obj = worker.graph
        if graph_obj is None:
            return []
        cp = getattr(graph_obj, "checkpointer", None)
        if not cp:
            return []
        if asyncio.iscoroutinefunction(getattr(cp, "list", None)):
            fut = worker.run_coro(cp.list(None))
            checkpoints = fut.result(timeout=timeout)
        else:
            checkpoints = cp.list(None)
        all_threads = set()
        for checkpoint in checkpoints or []:
            try:
                cfg = getattr(checkpoint, "config", {}) or {}
                tid = cfg.get("configurable", {}).get("thread_id")
                if tid:
                    all_threads.add(tid)
            except Exception:
                continue
        return list(all_threads)
    except FutureTimeout:
        return []
    except Exception:
        traceback.print_exc()
        return []

# --------------------------- AsyncWorker (single event loop in background thread) ---------------------------

class AsyncWorker:
    def __init__(self):
        self._thread = None
        self._loop = None
        self._started = threading.Event()
        self._graph = None

    def start(self, init_graph_coroutine, init_timeout=30):
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

# --------------------------- Streamlit app init ---------------------------
# create / start worker once per session
if 'async_worker' not in st.session_state or st.session_state['async_worker'] is None:
    worker = AsyncWorker()
    try:
        worker.start(init_graph)
    except Exception as e:
        st.error(f"Failed to start async worker: {e}")
        raise
    st.session_state['async_worker'] = worker
else:
    worker = st.session_state['async_worker']

def _cleanup():
    w = st.session_state.get('async_worker')
    if w:
        try:
            w.stop()
        except Exception:
            pass

atexit.register(_cleanup)

# --------------------------- Session state defaults ---------------------------
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads_from_worker(worker)

add_thread(st.session_state['thread_id'])

# --------------------------- Sidebar UI ---------------------------
st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

# list thread 
for thread_id in st.session_state['chat_threads'][::-1]:
    cols = st.sidebar.columns([3, 1])
    # select button
    if cols[0].button(str(thread_id), key=f"select_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation_sync_from_worker(worker, thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': getattr(msg, "content", str(msg))})

        st.session_state['message_history'] = temp_messages

    # delete button
    if cols[1].button("Delete", key=f"del_{thread_id}"):
        try:
            fut = worker.delete_thread(thread_id)
            ok = fut.result(timeout=10)
            if ok:
                st.success(f"Deleted thread {thread_id}")
                if thread_id in st.session_state['chat_threads']:
                    st.session_state['chat_threads'].remove(thread_id)
                    st.session_state['chat_threads'] = retrieve_all_threads_from_worker(worker)

                if st.session_state.get('thread_id') == thread_id:
                    st.session_state['thread_id'] = generate_thread_id()
                    st.session_state['message_history'] = []
            else:
                st.error(f"Failed to delete thread {thread_id}")
        except Exception as e:
            st.error(f"Error while deleting thread: {e}")
            traceback.print_exc()

# --------------------------- Main UI ---------------------------
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    q = queue.Queue()

    try:
        future = worker.start_astream_task({'messages': [HumanMessage(content=user_input)]}, CONFIG, q)
    except Exception as e:
        st.error(f"Failed to start stream task: {e}")
        future = None
        q.put(None)

    with st.chat_message('assistant'):
        container = st.empty()
        accumulated = ""
        while True:
            chunk = q.get()
            if chunk is None:
                break
            if isinstance(chunk, str) and chunk.startswith("[STREAM ERROR]"):
                container.error(chunk)
                try:
                    graph_obj = worker.graph
                    if graph_obj and hasattr(graph_obj, "ainvoke"):
                        fut = worker.run_coro(graph_obj.ainvoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG))
                        res = fut.result(timeout=10)
                        accumulated = getattr(res, "content", str(res)) if not isinstance(res, dict) else res.get("final_answer", str(res))
                    else:
                        accumulated = "Xin lỗi, hệ thống gặp sự cố."
                except Exception as exc:
                    traceback.print_exc()
                    accumulated = "Xin lỗi, hệ thống gặp sự cố."
                break

            accumulated += chunk
            container.text(accumulated)

    st.session_state['message_history'].append({'role': 'assistant', 'content': accumulated})