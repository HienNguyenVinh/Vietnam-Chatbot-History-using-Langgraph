from typing import List, Any, Dict, TypedDict, cast
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langgraph.graph import START, END, StateGraph
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from mxbai_rerank import MxbaiRerankV2
import asyncio
import logging
from dotenv import load_dotenv
import chromadb
from langchain_community.retrievers import BM25Retriever
from ..utils.utils import config
from .states import State
from .prompts import GENERATE_QUERY_SYSTEM_PROMPT

load_dotenv()

EMBEDDING_MODEL = config["retriever"]["embedding_model"]
RERANK_MODEL = config["retriever"]["rerank_model"]
GENERATE_QUERY_MODEL = config["retriever"]["generate_query_model"]

TOP_K = config["retriever"]["top_k"]
COLLECTION_NAME = config["retriever"]["collection_name"]
DB_PATH = config["retriever"]["db_path"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Query(TypedDict):
    vector_search_query : str
    category: str

    bm25_search_keyword: str

async def generate_query(state: State) -> Dict[str, str]:
    logger.info("___generating queries...")
    model = ChatGoogleGenerativeAI(model=GENERATE_QUERY_MODEL)
    messages=[
        {"role": "system", "content": GENERATE_QUERY_SYSTEM_PROMPT},
        {"role": "human", "content": state.user_query}
    ]

    response = cast(Query, await model.with_structured_output(Query).ainvoke(messages))
    logger.info(f"___generated queries: {response}")

    return response

async def vector_search(query: str, category: str) -> List[Document]:
    client = PersistentClient(path=DB_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        model = SentenceTransformer(EMBEDDING_MODEL)

        embedding = model.encode(query, convert_to_numpy=True)
        include = ["metadatas", "documents", "distances"]

        where = {}
        where["category"] = category

        results = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=TOP_K,
            where=where,
            include=include
        )

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0] if "ids" in results else [None] * len(docs)

        documents: List[Document] = []
        for _id, doc_text, md, dist in zip(ids, docs, metadatas, distances):
            metadata = {}
            if isinstance(md, dict):
                metadata.update(md)
            else:
                metadata["raw_metadata"] = md

            if _id is not None:
                metadata.setdefault("id", _id)
            metadata["distance"] = dist
            doc = Document(page_content=doc_text, metadata=metadata)

            documents.append(doc)

        return documents

    finally:
        try:
            client.persist()
        except Exception:
            pass

# Viết hàm bm25 search
# Dùng thư viện from langchain_community.retrievers import BM25Retriever
# Ec ec ec
async def bm25_search(bm25_search_keyword: str) -> List[Document]:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    raw = collection.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(raw["documents"], raw["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(all_docs, k=TOP_K)
    return bm25_retriever.ainvoke(bm25_search_keyword)

async def hybrid_search(state: State) -> Dict[Any, Any]:
    logger.info("___start searching...")
    results = await asyncio.gather(
        vector_search(state.vector_search_query, state.category),
        bm25_search(state.bm25_search_keyword),
        return_exceptions=True
    )
    logger.info("___finished searching...")

    vector_results, bm25_results = results
    if isinstance(vector_results, Exception):
        logger.error("Vector search failed", exc_info=vector_results)
        vector_results = []
    if isinstance(bm25_results, Exception):
        logger.error("Full-text search failed", exc_info=bm25_results)
        bm25_results = []

    logger.info(f"___vector search results: {len(vector_results)}...")
    logger.info(f"___bm25 search results: {len(bm25_results)}...")
    seen = set()
    combined: List[Any] = []
    for doc in vector_results + bm25_results:
        uid = doc.metadata.get("relative_path")
        if uid and uid not in seen:
            combined.append(doc)
            seen.add(uid)
    # print(combined)
    return {"retrieved_documents": combined}

def _format_documents(documents: List[Document]):
    results = []
    for doc in documents:
        results.append(doc.page_content)
    
    return results

async def rerank(state: State) -> Dict[str, List[any]]:
    logger.info("___start reranking...")
    model = MxbaiRerankV2(RERANK_MODEL)
    query = state.user_query
    documents = _format_documents(state.retrieved_documents)

    try: 
        results = model.rerank(query, documents, return_documents=True, top_k=5)
    except Exception as e:
        results = state.retrieved_documents

    logger.info("___finished reranking...")
    print(results)
    return {"retrieved_documents": results}

builder = StateGraph(State)

builder.add_node("generate_query", generate_query)
builder.add_node("hybrid_search", hybrid_search)
builder.add_node("rerank", rerank)

builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "hybrid_search")
builder.add_edge("hybrid_search", "rerank")
builder.add_edge("rerank", END)

graph = builder.compile()


if __name__ == '__main__':
    result = graph.invoke({"user_query": "Hồ Quý Ly đã làm gì?"})
    print(result['output'])


# [
#     {'id': 'Lich_Su_Chung__preprocessed_tap_3.txt__319__b2fb657c', 
#      'document': '- Năm 1576 (niên hiệu Vạn Lịch thứ 4), triều Mạc một lần cử sứ bộ sang triều Minh nộp cống.\n- Ngày Giáp Thìn tháng Giêng mùa Xuân năm Vạn Lịch thứ 4 (1576), An Nam Đô thống sứ Mạc Mậu Hợp sai Tuyên phủ đồng trì Lê Như Hỗ dẫn 73 nhân viên sang nộp cống. Thượng quốc sai đón tiếp, ban yến thưởng như lệ thường”.\n~ Năm 1581, triều Mạc nộp đủ cho nhà Minh 4 kỳ cống phẩm còn thiếu của các năm 1557 (Gia Tĩnh thứ 36), đời Mạc Phúc Nguyên; 1560 (Gia Tĩnh thứ 39), đời Mạc Phúc Nguyên và 2 lần dâng đặc sản địa phương chưa thực hiện của các năm 1575 (Vạn Lịch thứ 3), đời Mạc Mậu Hợp; 1578 (Vạn Lịch thứ 6), đời Mạc Mậu Hợp.\nMinh Thân Tông thực lục, ghi: "Ngày Tân Hợi tháng Sáu năm Vạn Lịch thứ 9 (1581), An Nam Đô thống sứ Mạc Mậu Hợp sai Tuyên phủ đồng tri Lương Phùng Thời dâng tờ biểu về bỏ khuyết cống tuế vào các năm Gia Tĩnh thứ 36, Gia Tĩnh thử 39, đây là năm cống chính, còn năm Vạn Lịch thứ 3 và Vạn Lịch thứ 6 là năm đâng phương vật. Bộ Lễ đáp rằng: "Mậu Hợp bổ sung cả 4 lễ cống, chứng tỏ lòng trung thành quy thuận, thật đáng khen".\nChiếu cho ban thưởng yến tiệc và cấp sắc phong "2.\nĐến thời Mạc Mậu Hợp, triểu Mạc đã đi vào giai đoạn suy vong. trước sự phản công mạnh mẽ của lực lượng triều Lê. Việc triều Mạc liên tiếp cống nạp của cải cho triều Minh chính là muốn dựa vào triều Minh làm hậu thuẫn giúp đỡ nhằm mục đích duy trì, bảo vệ quyền lợi riêng của dòng họ Mạc khi bị triều Lê đánh bại.\nTrong quan hệ chính trị giữa triều Mạc và triều Minh, có một sự kiện đáng chú ý là vấn đẻ Phạm Tử Nghĩ.\nNăm 1546, khi Mạc Phúc Hải chết, Mạc Phúc Nguyên là con trưởng được kế vị. Nhưng Tứ Dương hầu Phạm Từ Nghỉ, một tướng của triều Mạc, lại mưu lập Mạc Chính Trung. con thứ của Mạc Đăng Dung. Việc không thành, Phạm Tử Nghi đưa Chính Trung về xã Hoa Dương, huyện Ngự Thiên\'. Theo Đại Việt sử ký toàn thư, họ Mạc sai Khiêm vương là Kính cùng với Tây quận công là Nguyễn Kính đem quân đi bắt, bị Tử Nghỉ đánh cho thua. Sau Tử Nghỉ máy lần đánh không được, mới đem Chính Trung ra chiếm cứ miền Yên Quảng. Dân hạt Hải Dương bị nạn binh lửa nhiều, nhiều người phải lưu vong. Tử Nghỉ lại trốn vào đất nước Minh, cho quân đi bắt người cướp của ở Quảng Đông, Quảng Tây, người Minh không thể kiềm chế được.', 
#      'metadata': {'category': 'Lich_Su_Chung', 'relative_path': 'Lich_Su_Chung/preprocessed_tap_3.txt', 'chunk_index': 319, 'file': 'preprocessed_tap_3.txt'}, 
#      'distance': 0.9747435450553894}, 
#     {'id': 'Lich_Su_Chung__preprocessed_tap_2.txt__278__8056d313', 
#      'document': 'Có lần sứ nhà Nguyên sang Đại Việt báo việc lên ngôi nhưng tò ra rất ngạo mạn không tuân theo quy định cùa Đại Việt “Năm 1324, nhà Nguyên sai bọn Thượng thư Mã Hợp Mưu và Dương Tông Thụy sang báo việc lên ngôi và cho một quyển lịch. Vua sai Mạc Đĩnh Chi sang mừng"3.\nTiếc rằng, các sách sử của nước ta chi chép sự kiện sứ nhà Nguyên sang nước ta, đi lại rất ngông nghênh mà chép quá sơ sài lần đi sứ này của Mạc Đĩnh Chi nên không thể kể rõ sự việc. Toàn thư chép: "Năm Giáp Tý (1324), tháng 4... Vua Nguyên sai Mã Hợp Mưu và Dương Tông Thụy sang báo việc lên ngôi và cho một quyển lịch. Bọn Hợp Mưu cưỡi ngựa đến tận đường ở cầu Tây Thấu trì không xuống. Những người biết nói tiếng Hán vâng chi đến tiếp chuyện, từ giờ Thìn đến giờ Ngọ, vẻ giận càng hăng. Vua sai Thị ngự sử Nguyễn Trung Ngạn ra đón. Trung Ngạn lấy lời lẽ bẻ lại, Hợp Mưu không cãi được, phải xuống ngựa, bưng tờ chiếu đi bộ"4.\nVà, không có một dòng nào chép về việc Mạc Đĩnh Chi đi sứ. Sách Cưomg mục, Đại Việt sử ký tiền biên đã bổ sung thêm chi tiết đó: "Bọn Hợp Mưu trở về vua sai Mạc Đĩnh Chi sang chúc mừng"5.\nChính sử của triều Nguyên ghi lại như sau: "Năm thứ nhất niên hiệu Thái Định (1324), Thế tử Trần Nhật Khoáng (tức vua Trần Minh Tông) sai bề tôi là bọn Mạc Tiết Phu đến tiến cống"1. Và, tiếng tăm của lần đi sú lần trước đã khiến cho chuyến đi sứ lần này của Mạc Đĩnh Chi được các sử thần triều Nguyên trân trọng ghi chép trong chính sử Trung Quốc...2.\nTrong thời gian tiếp theo, nội dung bang giao giữa hai nước cũng chủ yếu là những thông báo mang tính chất nghi lễ lên ngôi và chúc mừng. Ví dụ: nhà Nguyên sai Lại bộ Thượng thư Tát Chí Ngõa sang báo việc lên ngôi. Vua sai Đoàn Tử Trinh sang cống và mừng lên ngôi3.\nNăm 1345, Toàn thư chép sự kiện sứ nhà Nguyên là Vương Sĩ Hành sang hỏi về việc cột đồng. Vua Trần đã sai Phạm Sư Mạnh sang Nguyên để biện bạch4.\nThời gian này, ở nước Nguyên giặc cướp nổi lên khắp nơi.\nTrong hoàn cảnh đó, nhà Nguyên phải lo dẹp loạn nên việc bang giao chi mang tính chất nghi lễ, không còn khả năng vừa dùng ngoại giao vừa thăm dò và thám thính Đại Việt như trước.\n1. Xem thêm: Nguyễn Hữu Tâm, Mạc Đĩnh Chi vói hai lần đi sứ, bản thào.\nnhau với Trần Hữu Lượng, chưa phân được thua. Vua sai Lê Kính Phu sang sứ phương Bắc để xem hư thực thế nào”1.\nNhưng cuộc chiến của Minh Thái Tổ với Trần Hữu Lượng vẫn chưa phân thắng bại nên nhà Minh yêu cầu Đại Việt cung ứng quân lương nhưng đã bị vua Đại Việt khước từ. Sử chép: Năm 1361, Minh Thái Tổ đánh Giang Châu, Trần Hữu Lượng lui về giữ Vũ Xương. Minh Thái Tổ sai ngirời sang Đại Việt yêu cầu cung cấp quân cứu viện nhưng vua Trần đã từ chối.', 
#      'metadata': {'relative_path': 'Lich_Su_Chung/preprocessed_tap_2.txt', 'chunk_index': 278, 'category': 'Lich_Su_Chung', 'file': 'preprocessed_tap_2.txt'}, 
#      'distance': 1.0432504415512085}
# ]