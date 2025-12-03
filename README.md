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
- Unzip file chroma_db.zip ở ngoài thư mục chính (cùng cấp với thư mục src)
- Vào thư mục chroma_db (1) sau khi giải nén, vào tiếp trong thư mục content
- Sẽ có 1 thư mục chroma_db (2) trong thư mục content
- Lấy hết các file trong thư mục chroma_db (2) và vất vào trong thư mục chroma_db (1)
- Xóa thư mục content rỗng đi
- Chạy file test_db để test db
```
   cd src/db_helper
   python test_db.py
   cd ..
```

### Tạo file .env với nội dung như sau:
GOOGLE_API_KEY=""
TAVILY_API_KEY=""
HF_TOKEN=""

DB_NAME=""
DB_USER=""
DB_PASSWORD=""
DB_HOST="localhost"
DB_PORT="5432"
