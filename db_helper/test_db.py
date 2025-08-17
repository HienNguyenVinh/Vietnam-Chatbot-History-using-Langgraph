import numpy as np
import chromadb
from typing import Optional, Any, Dict, List
from sentence_transformers import SentenceTransformer
from pathlib import Path

def vector_search_chroma(
    collection,
    query: Optional[str] = None,
    query_embedding: Optional[Any] = None,
    embedder: Optional[Any] = None,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Thực hiện vector search trên chromadb collection.

    Args:
      collection: chromadb collection (client.get_or_create_collection(...))
      query: (optional) câu truy vấn dạng text (string). Nếu truyền query_embedding thì query sẽ bị bỏ qua.
      query_embedding: (optional) embedding đã tính sẵn (1-d array / list).
      embedder: (optional) object có method `encode(text, convert_to_numpy=True)` hoặc callable trả embedding.
                Nếu query được cung cấp và query_embedding is None thì embedder bắt buộc.
      top_k: số kết quả trả về cho mỗi truy vấn.
      where: dict metadata filter (ví dụ {"category": "Chinh_Tri"}).
      include: list fields muốn lấy từ Chroma (ví dụ ["metadatas","documents","distances","ids"]).
               Mặc định sẽ lấy metadatas, documents, distances, ids.
      score_threshold: nếu set thì chỉ trả về kết quả có distance <= threshold (tuỳ distance metric của collection).

    Returns:
      list of dict: [{'id':..., 'document':..., 'metadata':..., 'distance':...}, ...]
    """

    if query_embedding is None and query is None:
        raise ValueError("Bạn phải truyền query (text) hoặc query_embedding (vector).")

    # compute embedding if necessary
    if query_embedding is None:
        if embedder is None:
            raise ValueError("Khi truyền query text, bạn phải cung cấp embedder.")
        # embedder có thể là SentenceTransformer hoặc callable
        if hasattr(embedder, "encode"):
            q_emb = embedder.encode(query, convert_to_numpy=True)
        else:
            q_emb = embedder(query)
    else:
        q_emb = query_embedding

    # normalize to list
    q_emb = np.asarray(q_emb)
    if q_emb.ndim == 1:
        q_emb_list = q_emb.tolist()
    else:
        # nếu vô tình truyền nhiều embedding, lấy embedding đầu tiên
        q_emb_list = q_emb[0].tolist()

    # default include fields
    if include is None:
        include = ["metadatas", "documents", "distances"]

    # call chroma
    try:
        res = collection.query(
            query_embeddings=[q_emb_list],
            n_results=top_k,
            where=where,
            include=include
        )
    except Exception as e:
        raise RuntimeError(f"Error while querying Chroma: {e}")

    # Chroma trả về dict với các lists (mỗi query một entry)
    docs = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]
    ids = res.get("ids", [[]])[0] if "ids" in res else [None] * len(docs)

    results = []
    for _id, doc, md, dist in zip(ids, docs, metadatas, distances):
        if score_threshold is not None:
            # chú ý: khoảng cách (distance) nhỏ hơn tốt hơn (tuỳ metric). nếu bạn dùng similarity thay đổi điều kiện.
            if dist is None:
                pass
            else:
                try:
                    if dist > score_threshold:
                        continue
                except Exception:
                    pass
        results.append({
            "id": _id,
            "document": doc,
            "metadata": md,
            "distance": dist
        })
    return results

print("___________Start_____________")
client = chromadb.PersistentClient(path=Path(__file__).parent / "chroma_db")
print(client.list_collections())
collection = client.get_collection("viet_history")
print("___________get db success__________")

embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
print("___________get embedding model success_________")

print("___________searching____________")
query = "Năm 1304, có sự kiện gì xảy ra với Mạc Đĩnh Chi?"
results = vector_search_chroma(
    collection=collection,
    query=query,
    embedder=embedder,
    top_k=5,
    where={"category": "Lich_Su_Chung"}
)

print(type(results))
for i, r in enumerate(results, 1):
    print(i, "distance:", r["distance"])
    print("file:", r["metadata"].get("file"), "chunk_index:", r["metadata"].get("chunk_index"))
    print(r["document"][:400].replace("\n", " "), "...\n")

print(results)
