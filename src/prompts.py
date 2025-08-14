CLASSIFIER_SYSTEM_PROMPT = """Phân loại câu hỏi sau đây. Nếu có liên quan đến lịch sử từ năm 2000 trở lên, trả lời: web. Nếu từ năm 1999 trở về trước, trả lời: db. Không cần giải thích."""

RESPONSE_SYSTEM_PROMPT = """
Dựa trên kết quả dưới đây, hãy trả lời truy vấn một cách ngắn gọn và chính xác.
"""

REFLECTION_PROMPT = """
Bạn là một AI kiểm duyệt. Hãy đánh giá câu trả lời sau có đủ tốt không (good hoặc bad). Nếu tốt thì giữ nguyên. Nếu chưa tốt, hãy đề xuất tìm kiếm lại.
"""