import os
import dotenv
import psycopg
from psycopg.rows import dict_row

dotenv.load_dotenv()

db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")

def get_db_connection():
    return psycopg.connect(
        host=db_host,
        dbname=db_name,
        user=db_user,
        password=db_password,
        port=db_port,
        row_factory=dict_row
    )

if __name__ == "__main__":
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                print(cur.fetchone())
    except Exception as e:
        print("Lỗi khi test kết nối:", e)