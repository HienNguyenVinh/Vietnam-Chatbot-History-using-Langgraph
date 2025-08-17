from utils.utils import config, PATH_DB
from src.sub_graph.states import State
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from src.sub_graph.prompts import TEMPLATE_PROMPT_GEN_QUERY

load_dotenv()

# Tên model embedding được lấy từ file config
EMBEDDING_MODEL = config["retriever"]["embedding_model"]

# Số lượng tài liệu gần nhất sẽ lấy khi truy vấn trong vector DB
TOP_K = config["retriever"]["top_k"]

# Tên collection trong ChromaDB
COLLECTION_NAME = config["retriever"]["collection_name"]

# Tên model chat (Gemini) để sử dụng làm LLM trong Google Generative AI
MODEL_CHAT_NAME = config['llm']['gemini']

# API key của Google Generative AI, được load từ biến môi trường (.env)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


def llm():
    chat = ChatGoogleGenerativeAI(model=MODEL_CHAT_NAME, google_api_key=GOOGLE_API_KEY)
    return chat
    pass


def start(data: State):
    return data
    pass


def generate_query(data: State):
    '''
    Sinh propmt đầy đủ cho phiên hiện tại bằng cách kết hợp với lịch sử chat
    Args:
        data: Gồm câu hỏi người dùng và lịch sử trò truyện

    Returns: Câu hỏi độc lập
    '''
    chat = llm()  # Khởi tạo LLM (Google Generative AI) với model đã config
    input = data['input']  # Lấy câu hỏi đầu vào từ người dùng
    history = data.get('history', [])  # Lấy lịch sử hội thoại, nếu chưa có thì dùng list rỗng
    # Tạo PromptTemplate với biến {question} và template có sẵn
    prompt_template = PromptTemplate(
        input_variables=['question'],
        template=TEMPLATE_PROMPT_GEN_QUERY
    )
    # Format template bằng input người dùng → sinh ra câu prompt cuối cùng
    end_prompt = prompt_template.format(question=input)
    # Thêm câu hỏi (đã format) vào lịch sử như một HumanMessage
    history.append(HumanMessage(content=end_prompt))
    # Gọi mô hình LLM với toàn bộ lịch sử → sinh ra AIMessage
    ai_message = chat.invoke(history)
    # Thêm phản hồi của AI vào lịch sử
    history.append(ai_message)
    # Trả về query (chuỗi độc lập) và lịch sử đã được cập nhật
    return {
        'query': ai_message.content,
        'history': history,
    }
    pass


def retrieval(data: State):
    """
    Thực hiện bước truy xuất trong RAG

    Args:
        data (State): Gồm câu truy vấn độc lập (string)

    Returns: Kết quả trả về từ ChromaDB (bao gồm documents, ids, distances, ...)
    """
    # Kết nối đến vector database (ChromaDB) tại đường dẫn PATH_DB
    client = PersistentClient(path=PATH_DB)
    # Lấy ra collection (bảng chứa dữ liệu vector hoá) theo tên đã config
    collection = client.get_collection(COLLECTION_NAME)
    # Load sentence transformer model để tạo embedding cho câu truy vấn
    model = SentenceTransformer(EMBEDDING_MODEL)
    # Biến câu query thành vector embedding (dạng numpy array)
    embedding = model.encode([data['query']], convert_to_numpy=True)
    # Truy vấn vào collection với embedding vừa tạo
    result = collection.query(
        query_embeddings=embedding.tolist(),  # chuyển numpy array → list
        n_results=TOP_K  # Số lượng vector gần nhất(so với suy vấn) cần lấy
    )
    print(result['distances'])
    # Trả về kết quả tìm kiếm (result chứa documents, embeddings, distances, ids...)
    return {
        'result': result
    }
    pass


def rerank(data: State):
    documents = data['result']['documents']
    distances = data['result']['distances']
    best_index = distances.index(min(distances))
    return {'output': documents[best_index]}
    pass


def end(data: State):
    return {'output': data['output'][0]}
    pass


# Khởi tạo trình dụng
builder = StateGraph(State)

# Thêm các node cho trình dựng
builder.add_node("start", start)
builder.add_node("generate_query", generate_query)
builder.add_node("retrieval", retrieval)
builder.add_node("rerank", rerank)
builder.add_node("end", end)

# Thêm các cạnh giữa các node trong trình dựng
builder.add_edge("start", "generate_query")
builder.add_edge("generate_query", "retrieval")
builder.add_edge("retrieval", "rerank")
builder.add_edge("rerank", "end")

# Setup điểm đầu cuối của trình dựng
builder.set_entry_point("start")
builder.set_finish_point("end")

# Compile trình dựng tạp graph
graph = builder.compile()

if __name__ == '__main__':
    result = graph.invoke(
        {"input": "Bạch Đằng"})  # Bên graph của Quang truyền vào là chuỗi chứ không phải dict nên chưa khớp
    from pprint import pprint # In bằng pprint cho đẹp
    print(result)
