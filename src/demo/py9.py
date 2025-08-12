import os
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from typing import TypedDict, Literal
from ddgs import DDGS


# Khu vực khởi tạo các biến cục bộ tĩnh


model = 'models/gemini-2.5-flash'
embed_model = 'models/embedding-001'
path_file_data = r"D:\Python exflorer\Langchain\hoi_dap_tren_tai_lieu\lien_quan.txt"
api_key = os.getenv('GOOGLE_API_KEY')
prompt_template_agent = '''
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
'''


# Khu khu vực khai báo các hàm cơ bản


def chia_chunk_tai_lieu(path: str):
    loai_file = os.path.splitext(path)[1].lstrip('.').lower()
    if loai_file == 'pdf':
        trinh_tai_du_lieu = PyPDFLoader(path)
    elif loai_file == 'txt':
        trinh_tai_du_lieu = TextLoader(path, encoding='utf-8')
    else:
        raise ValueError("Chỉ hỗ trợ 'pdf' hoặc 'txt'")

    noi_dung = trinh_tai_du_lieu.load_and_split()
    noi_dung = [trang.page_content for trang in noi_dung]

    trinh_tach = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = trinh_tach.create_documents(noi_dung)
    return chunks

def tim_kiem(query: str) -> str:
    '''
    Hàm tìm kiếm thông tin bằng DuckDuckGoSearch
    :param query: Chuỗi truy vấn thông tin
    :return: Trả về kết quả được tổng hợp từ các trang web liên quan
    '''
    with DDGS() as ddgs:
        cac_ket_qua = ddgs.text(query=query, max_results=5)
        chuoi_tong_hop = ''
        for ket_qua in cac_ket_qua:
            chuoi_tong_hop += ket_qua['title'] + '\n' + ket_qua['href'] + '\n' + ket_qua['body'] + '\n\n'
        return chuoi_tong_hop

def rag(dau_vao: str):
    '''
    Hàm truy vấn thông tin trong cơ sở dữ liệu vector
    :param dau_vao: Câu hỏi truy vấn
    :return: Câu trả lời
    '''
    chunks = chia_chunk_tai_lieu(path_file_data)

    # Khởi tạo embedding và tạo vector database
    embeddings = GoogleGenerativeAIEmbeddings(model=embed_model)
    vector_db = FAISS.from_documents(chunks, embeddings)

    # Khởi tạo mô hình hỏi đáp
    retriever = vector_db.as_retriever()
    chat = ChatGoogleGenerativeAI(model='models/gemini-2.5-flash-lite-preview-06-17')
    qa_chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, chain_type="stuff")

    tra_loi = qa_chain.invoke(dau_vao)
    print(f"Câu trả lời: {tra_loi['result']}")
    return tra_loi['result']

def tao_executor() -> AgentExecutor:
    '''
    Tạo 1 executor cho agent AI
    :param ten_model:  Tên mô hình LLM được sử dụng
    :param api_key: API key của nền tảng LLM được sử dụng
    :return: AgentExecutor
    '''
    chat = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
    )
    tim_kiem_tool = Tool(
        name='tim_kiem_tool',
        func=tim_kiem,
        description='Dùng để tìm kiếm thông tin mới nhất từ web bằng DuckDuckGo'
    )
    rag_tool = Tool(
        name='rag_tool',
        func=rag,
        description='Dùng để tìm kiếm thông tin trong cơ sở dữ liệu vector'
    )
    tools = [tim_kiem_tool, rag_tool]
    prompt = PromptTemplate.from_template(prompt_template_agent)
    agent = create_react_agent(chat, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return executor


# Khu vực khai báo các hàm LangGraph


class TrangThai(TypedDict):
    dau_vao: str
    dau_ra: str


def nhan_dau_vao(du_lieu: TrangThai):
    du_lieu['dau_vao'] = input('- User input:')
    return du_lieu
    pass


def agent_tim_kiem(du_lieu: TrangThai):
    agent = tao_executor()
    ket_qua = agent.invoke({
        'input': du_lieu['dau_vao']
    })
    return {'dau_ra': ket_qua['output']}
    pass


def nhan_ket_qua_tam_thoi(du_lieu: TrangThai):
    print(du_lieu['dau_ra'])
    return du_lieu


if __name__ == '__main__':
    trinh_xay_dung = StateGraph(TrangThai)
    trinh_xay_dung.add_node('nhan_dau_vao', nhan_dau_vao)
    trinh_xay_dung.add_node('agent_tim_kiem', agent_tim_kiem)
    trinh_xay_dung.add_node('nhan_ket_qua_tam_thoi', nhan_ket_qua_tam_thoi)

    trinh_xay_dung.add_edge('nhan_dau_vao', 'agent_tim_kiem')
    trinh_xay_dung.add_edge('agent_tim_kiem', 'nhan_ket_qua_tam_thoi')

    trinh_xay_dung.set_entry_point('nhan_dau_vao')
    trinh_xay_dung.set_finish_point('nhan_ket_qua_tam_thoi')

    ung_dung = trinh_xay_dung.compile()
    ket_qua = ung_dung.invoke({})
    pass
