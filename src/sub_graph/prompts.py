TEMPLATE_PROMPT_GEN_QUERY = """
Dựa vào lịch sử trò chuyện sau đây và câu hỏi cuối cùng của người dùng,
hãy tạo ra một câu hỏi độc lập, đầy đủ ngữ cảnh để có thể dùng nó tìm kiếm thông tin mà không cần xem lại lịch sử.
Chỉ trả về câu hỏi đó, không thêm bất kỳ lời giải thích nào.
Câu hỏi cuối cùng: {question}
Câu hỏi độc lập:
"""