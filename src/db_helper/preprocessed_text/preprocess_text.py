import pdfplumber
import os
import re


def extract_text(file_path, first_page):
    # Chuẩn bị tên tệp đầu ra
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = "text"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"text_{base_name}.txt")

    full_text = ""
    with pdfplumber.open(file_path) as pdf:

        # for page in pdf.pages[first_page - 1:first_page+5]:
        for page in pdf.pages[first_page - 1:]:
            page_txt = page.extract_text() or ""
            lines = page_txt.split("\n")

            # Xóa số trang
            if lines and re.fullmatch(r"\s*\d+\s*", lines[-1]):
                lines.pop(-1)

            # Xóa header
            if lines and re.fullmatch(r" LỊCH SỬ VIỆT NAM - TẬP 7", lines[0], flags=re.IGNORECASE):
                lines.pop(0)

            cleaned_page = "\n".join(lines)
            full_text += cleaned_page + "\n\n"

    cleaned = re.sub(r'\n{3,}', '\n\n', full_text)         
    cleaned = re.sub(r'(?<![.?!])\n+', ' ', cleaned)       
    cleaned = re.sub(r'\n+', '\n', cleaned)
    cleaned = re.sub(r' LỊCH SỬ VIỆT NAM - TẬP 7', '', cleaned)   

    # Ghi ra file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned)

def fix_text(input_path):
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        full_text = f.read()
    
    lines = full_text.splitlines()
    def is_citation(line: str) -> bool:
        # hoặc nếu chứa các từ khóa xuất bản/chú thích phổ biến
        if re.search(r'\bNxb\.|\btr\.|\bSđd|\btờ\.|\bQ\.', line, flags=re.IGNORECASE):
            return True
        # hoặc nếu chỉ gồm dấu sao và whitespace
        if re.fullmatch(r'[\*\s\.,;:/&quot;]+', line):
            return True
        return False

    filtered_lines = [ln for ln in lines if not is_citation(ln)]
    cleaned = []
    for line in filtered_lines:
        # bỏ dòng chỉ toàn số (ví dụ "19\n")
        if re.match(r'^\s*\d+\s*$', line):
            continue
        # bỏ dòng là tiêu đề cố định
        if 'LỊCH SỬ VIỆT NAM - TẬP 7 ' in line:
            continue
        cleaned.append(line.strip())
    full_text = "\n".join(cleaned)

    fixed = re.sub(r'\n{3,}', '\n\n', full_text)         
    # fixed = re.sub(r'(?<![.?!])\n+', ' ', fixed)       
    # fixed = re.sub(r'\n+', '\n\n', fixed)
    fixed = re.sub(r'LỊCH SỬ VIỆT NAM - TẬP 7', '', fixed)  

    output_dir = "preprocessed_text\\text_tap_7.txt"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "text_tap_7.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(fixed)

if __name__ == "__main__":
    fix_text(r'C:\Users\admin\OneDrive\Desktop\Project_Intern_MISA\db_helper\preprocessed_text\Lich_Su_Chung\preprocessed_tap_7.txt')

def preprocess_txt(input_path: str) -> str:
    # 1) Đọc nguyên file
    with open(input_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    full_text = re.sub(r'<span[^>]*>.*?</span>', '', full_text, flags=re.IGNORECASE)

    # 2) Xóa header/footer (như trước)
    # title = ""
    # pattern_hf = rf'\b\d+\s*{re.escape(title)}\b|\b{re.escape(title)}\b'
    # full_text = re.sub(pattern_hf, '', full_text, flags=re.IGNORECASE)

    # 3) Split thành lines để lọc footnotes
    lines = full_text.splitlines()

    def is_citation(line: str) -> bool:
        # nếu line chứa bất kỳ thẻ <sup>…</sup> nào → coi là chú thích
        if '<sup>' in line.lower() and '</sup>' in line.lower():
            return True
        # hoặc nếu chứa các từ khóa xuất bản/chú thích phổ biến
        if re.search(r'\bNxb\.|\btr\.|\bSđd|\btờ\.|\bQ\.', line, flags=re.IGNORECASE):
            return True
        # hoặc nếu chỉ gồm dấu sao và whitespace
        if re.fullmatch(r'[\*\s\.,;:/&quot;]+', line):
            return True
        return False

    filtered_lines = [ln for ln in lines if not is_citation(ln)]

    cleaned = "\n".join(filtered_lines)
    # cleaned = re.sub(r'^\s*\n', '\n', cleaned, flags=re.MULTILINE)

    # 4) Chuẩn hóa paragraph như trước
    # cleaned = re.sub(r'\n{2,}', '\n', cleaned)           # gom >=3 newline → 2 newline
    # cleaned = re.sub(r'(?<![.?!])\n+', ' ', cleaned)       # nối newline đơn
    # cleaned = re.sub(r'\s+', ' ', cleaned)
    # cleaned = re.sub(r'\n+', '\n\n', cleaned)              # chuẩn hóa mọi newline

    return cleaned

def save_preprocessed(input_path: str, output_dir: str = "preprocessed_text") -> str:
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    # out_path = os.path.join(output_dir, f"preprocessed_{base}.txt")
    out_path = input_path
    result = preprocess_txt(input_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(result)
    return out_path

# if __name__ == "__main__":
#     for n in [1]:
#         input_file = f"preprocessed_text\\ChinhTri\\LichSuDang.txt"
#         out_file = save_preprocessed(input_file)
#         print(f"Đã lưu file tiền xử lý tại: {out_file}")
    

    # print("starting...")
    # file_path = r"preprocessed_text\preprocessed_text_LSVN_Tap3.txt"
    # first_page = 19

    # # extract_text(file_path, first_page)
    # fix_text(input_path=file_path)

    # print("Finish...")