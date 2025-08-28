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
from src.utils.async_worker import AsyncWorker

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
                return state.values.get('current_chat', [])
            except Exception:
                pass
        if hasattr(graph_obj, "get_state"):
            fut = worker.run_coro(graph_obj.get_state({'configurable': {'thread_id': thread_id}}))
            return fut.result(timeout=timeout).values.get('current_chat', [])
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
st.sidebar.title('Chatbot tra cứu lịch sử Việt Nam')

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

        temp_messages = messages
        # temp_messages = []
        # for msg in messages:
        #     if isinstance(msg, HumanMessage):
        #         role = 'user'
        #     else:
        #         role = 'assistant'
        #     temp_messages.append({'role': role, 'content': getattr(msg, "content", str(msg))})

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
        if message['role'] == 'assistant':
            st.markdown(message['content'])
        else:
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
            container.markdown(accumulated)

    st.session_state['message_history'].append({'role': 'assistant', 'content': accumulated})