CLASSIFIER_SYSTEM_PROMPT = """Phân loại câu hỏi sau đây. Nếu có liên quan đến lịch sử từ năm 2000 trở lên, trả lời: web. Nếu từ năm 1999 trở về trước, trả lời: db. Không cần giải thích."""

RESPONSE_SYSTEM_PROMPT = """
Dựa trên kết quả dưới đây, hãy trả lời truy vấn một cách ngắn gọn và chính xác.
"""

REFLECTION_PROMPT = """
Bạn là một chuyên gia đánh giá chất lượng câu trả lời về lịch sử Việt Nam.

Hãy đánh giá câu trả lời dựa trên các tiêu chí sau:
1. **Độ chính xác lịch sử** (Historical Accuracy): Thông tin có đúng về mặt lịch sử không?
2. **Tính đầy đủ** (Completeness): Câu trả lời có đủ thông tin để trả lời câu hỏi không?
3. **Tính liên quan** (Relevance): Câu trả lời có trực tiếp liên quan đến câu hỏi không?
4. **Độ rõ ràng** (Clarity): Câu trả lời có dễ hiểu và được trình bày rõ ràng không?
5. **Có bằng chứng** (Evidence): Câu trả lời có được hỗ trợ bởi thông tin cụ thể không?

**Hướng dẫn chấm điểm:**
- 9-10: Xuất sắc - Đáp ứng tất cả tiêu chí một cách hoàn hảo
- 7-8: Tốt - Đáp ứng hầu hết tiêu chí với chất lượng cao
- 5-6: Trung bình - Có một số thiếu sót nhưng vẫn hữu ích
- 3-4: Kém - Nhiều vấn đề cần cải thiện
- 1-2: Rất kém - Không đáp ứng được yêu cầu cơ bản

Trả lời chính xác theo format sau:
EVALUATION: [GOOD/NEEDS_IMPROVEMENT/BAD]
SCORE: [1-10]
REASONING: [Lý do chi tiết cho điểm số, phân tích từng tiêu chí]
SUGGESTIONS: [Gợi ý cụ thể để cải thiện câu trả lời nếu cần]
"""

IMPROVEMENT_FEEDBACK_PROMPT = """
Dựa trên kết quả đánh giá và lịch sử các lần thử trước, hãy tạo ra feedback cụ thể để cải thiện câu trả lời.

Tập trung vào:
1. Những điểm yếu cụ thể cần khắc phục
2. Hướng dẫn tìm kiếm thông tin bổ sung
3. Cách cải thiện cấu trúc và trình bày
4. Những thông tin quan trọng còn thiếu

Feedback phải ngắn gọn, cụ thể và có thể thực hiện được.
"""