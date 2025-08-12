# README.md — Chatbot Tra cứu Lịch sử Việt Nam (Langgraph)

## Cài đặt
### Clone repo về máy
```
git clone https://github.com/HienNguyenVinh/Vietnam-Chatbot-History-using-Langgraph
```
### Cài thư viện cần thiết:
-  Nếu dùng pip, chạy lệnh sau trong terminal:
```python
   pip install -r requirements.txt
```
-  Nếu dùng uv, chạy lệnh sau trong terminal:
```python
   uv init
```

### Extract database:
- Unzip file chroma_db.zip trong file db_helper
- Vào thư mục chroma_db (1) sau khi giải nén, vào tiếp trong thư mục content
- Sẽ có 1 thư mục chroma_db (2) trong thư mục content
- Lấy hết các file trong thư mục chroma_db (2) và vất vào trong thư mục chroma_db (1)
- Xóa thư mục content rỗng đi
- Chạy file vector_search để test db


## 1. Quy tắc đặt tên

* **Ngôn ngữ:** Tiếng Anh cho tên biến/hàm/class/module (tránh tiếng Việt có dấu) trừ khi là nội dung hiển thị.
* **Biến (variables):** `snake_case` (ví dụ: `event_date`, `search_query`).

  * Biến boolean bắt đầu bằng `is_`, `has_`, `should_` (ví dụ: `is_verified`, `has_sources`).
* **Hàm (functions):** `snake_case` (ví dụ: `vector_search`, `fts_search`).
* **Lớp (classes):** `PascalCase` (ví dụ: `HistorySearcher`, `LanggraphNode`).
* **Hằng số (constants):** `UPPER_SNAKE` (ví dụ: `GOOGLE_API_KEY = 10`).
* **Tên file/module:** `snake_case.py` (ví dụ: `search_utils.py`).

---

## 2. Quy tắc viết hàm

* **Một nhiệm vụ duy nhất (SRP):** Mỗi hàm thực hiện đúng một việc. Nếu thấy nhiều trách nhiệm, tách nhỏ.
* **Độ dài:** Ưu tiên ngắn gọn — \~20–50 dòng là lý tưởng; nếu dài hơn, tách hàm con.
* **Type hints:** Luôn dùng type hints cho tham số và kiểu trả về (ví dụ: `def fetch_events(query: str, limit: int = 10) -> list[dict]:`).
* **Docstring:** Mỗi hàm public phải có docstring ngắn (1 dòng tóm tắt + chi tiết args/returns nếu cần). Dùng style Google hoặc NumPy.
* **Validation & lỗi:** Kiểm tra tham số đầu vào sớm (fail fast). Ném exception cụ thể (không dùng `Exception` chung), hoặc trả về lỗi theo chuẩn dự án.
* **Log & tracing:** Ghi log ở mức phù hợp (debug/info/warn/error).
* **Async vs Sync:** Node/Action tương tác IO (DB, API, file) nên khai báo `async def` nếu Langgraph flow dùng async. Giữ interface rõ ràng: async hàm trả `await` đúng chỗ.
* **Ví dụ ngắn:**

```python
async def vector_search(query: str, limit: int = 5) -> dict[str, Any]:
    """Tìm và trả về danh sách sự kiện khớp với query.

    Args:
        query: chuỗi tìm kiếm.
        limit: số lượng tối đa.

    Returns:
        dict chứa kết quả tìm kiếm
    """
    # validate
    if not query:
        raise ValueError("query is required")
    # implementation...
```

---

## 3. Quy tắc comment khi code

* **Docstrings trước tiên:** Dùng docstring cho module, lớp, hàm public. Trình bày mục đích, tham số, giá trị trả về, exception, ví dụ nếu cần.
* **Comment nội tuyến (inline):** Dùng cho "why" (tại sao làm thế này), không dùng để mô tả "what" khi code đã rõ.

  * Tốt: `# Use binary search to keep O(log n) behavior here` (giải thích lý do).
  * Không tốt: `# increment i by 1` (lặp lại những gì code đã thể hiện).
* **Cập nhật comment:** Khi sửa code, cập nhật comment docstrings luôn — comment lỗi thời gây nhầm lẫn.

---
