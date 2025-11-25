from .db_connection import get_db_connection
from typing import Dict, Optional, List
import json

def creat_db_chat_history_table():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Tạo UUID cho mỗi tin nhắn
                cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")

                # Tạo bảng
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS message (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        thread_id VARCHAR(255) NOT NULL,
                        user_question TEXT NOT NULL,
                        bot_answer TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Tạo index để tăng tốc truy vấn tin nhắn
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_message_thread_id 
                    ON message(thread_id)
                """)
                conn.commit()
    except Exception as error:
        print(f"Lỗi khi tạo bảng: {error}")

def clear_chat_history(thread_id: Optional[str] = None):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                if thread_id:
                    cursor.execute("DELETE FROM message WHERE thread_id = %s;", (thread_id,))
                else:
                    cursor.execute("TRUNCATE TABLE message;")
                conn.commit()
                print(f"Đã xóa {'tất cả' if not thread_id else 'thread ' + thread_id} dữ liệu trong bảng message.")
    except Exception as error:
        print(f"Lỗi khi xóa dữ liệu: {error}")

def save_message(thread_id: str, user_question: str, bot_answer: str) -> Optional[Dict]:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO message (thread_id, user_id, user_question, bot_answer) VALUES (%s, %s, %s, %s, %s) RETURNING id::text",
                    (thread_id, user_question, bot_answer)
                )
                result = cursor.fetchone()
                if result:
                    return result['id']
                return None
    except Exception as error:
        print(f"Lỗi khi lưu tin nhắn: {error}")
        return None

def get_chat_history(thread_id: str, limit: int=20) -> Optional[Dict]:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT thread_id, user_question, bot_answer, created_at 
                    FROM message 
                    WHERE thread_id = %s 
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (thread_id, limit)
                )
                results = cursor.fetchall()
                if results:
                    return results
                return None
    except Exception as error:
        print(f"Lỗi khi lấy lịch sử trò chuyện: {error}")
        return None
    
def format_chat_history(chat_history: List[Dict]) -> Optional[List]:
    if chat_history:
        formatted_history = []
        for message in reversed(chat_history):
            formatted_history.append({"role": "user",
                                      "content": message["user_question"]})
            formatted_history.append({"role": "model",
                                      "content": message["bot_answer"]})
    else:
        return []

    return formatted_history
    