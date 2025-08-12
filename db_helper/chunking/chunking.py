import os
import tempfile
import uuid
from typing import List
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
MAX_CHUNKING_TOKENS = 1024
MIN_CHUNKING_TOKENS = 128

DATA_DIR = "preprocessed_text"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "viet_history"
BATCH_SIZE = 128

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
def convert_txt_via_md(path, converter):
    content = load_txt(path)
    title = os.path.splitext(os.path.basename(path))[0]
    md_content = f"# {title}\n\n" + content

    # tạo file .md tạm thời
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp:
        tmp.write(md_content)
        tmp_path = tmp.name

    try:
        doc = converter.convert(source=tmp_path).document
    finally:
        os.remove(tmp_path)
    return doc

def infer_category(root_dir: str, file_path: str) -> str:
    # category is the top-level folder under root_dir (e.g. ChinhTri, Con_Nguoi, ...)
    rel = os.path.relpath(file_path, root_dir)
    parts = rel.split(os.path.sep)
    if len(parts) >= 2:
        return parts[0]
    return "root"


def chunk_file_with_docling(path: str, converter: DocumentConverter, chunker: HybridChunker):
    try:
        if path.endswith(".txt"):
            doc = convert_txt_via_md(path=path, converter=converter)
            chunks = list(chunker.chunk(dl_doc=doc))
            return chunks
    except Exception:
        text = load_txt(path)
        class SimpleChunk:
            def __init__(self, text):
                self.text = text
                self.metadata = {}
        return [SimpleChunk(text)]


def main(data_dir: str, chroma_dir: str, collection_name: str, max_batch: int):
    print("Starting...")
    converter = DocumentConverter()
    chunker = HybridChunker(
        max_tokens=MAX_CHUNKING_TOKENS, 
        min_tokens=MIN_CHUNKING_TOKENS, 
        overlap_tokens=100, 
        merge_peers=True
    )
    print("Init chunker finished...")

    # init embedder
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("Init embedding model finished...")

    # init chroma client
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=collection_name)
    print("Create chromaDB finished")

    all_docs: List[str] = []
    all_metadatas: List[dict] = []
    all_ids: List[str] = []

    files_count = 0
    chunks_count = 0

    print("Starting chunking and save...")
    for root, dirs, files in os.walk(data_dir):
        print(root)
        for fn in files:
            if not fn.lower().endswith('.txt'):
                continue
            files_count += 1
            fullpath = os.path.join(root, fn)
            relpath = os.path.relpath(fullpath, data_dir)
            category = infer_category(data_dir, fullpath)

            # chunk using docling
            chunks = chunk_file_with_docling(fullpath, converter, chunker)

            for i, chunk in enumerate(chunks):
                text = getattr(chunk, "text", None) or (chunk.get("text") if isinstance(chunk, dict) else None)
                metadata = getattr(chunk, "metadata", {}) if not isinstance(chunk, dict) else chunk.get("metadata", {})
                if text is None:
                    # if chunk is a dict
                    text = chunk if isinstance(chunk, str) else str(chunk)

                # enrich metadata
                md = dict(metadata) if metadata else {}
                md.update({
                    "category": category,
                    "file": fn,
                    "relative_path": relpath,
                    "chunk_index": i,
                })

                uid = f"{relpath.replace(os.path.sep, '__')}__{i}__{uuid.uuid4().hex[:8]}"

                all_docs.append(text)
                all_metadatas.append(md)
                all_ids.append(uid)
                chunks_count += 1

                # flush in batches to avoid huge memory
                if len(all_docs) >= max_batch:
                    print(f"Embedding & adding batch of {len(all_docs)} chunks to Chroma...")
                    embs = embedder.encode(all_docs, convert_to_numpy=True, show_progress_bar=False)
                    collection.add(documents=all_docs, metadatas=all_metadatas, ids=all_ids, embeddings=embs.tolist())
                    all_docs.clear(); all_metadatas.clear(); all_ids.clear()

    # final flush
    if all_docs:
        print(f"Final embedding & adding batch of {len(all_docs)} chunks to Chroma...")
        embs = embedder.encode(all_docs, convert_to_numpy=True, show_progress_bar=False)
        collection.add(documents=all_docs, metadatas=all_metadatas, ids=all_ids, embeddings=embs.tolist())

    # persist DB
    try:
        client.persist()
    except Exception:
        # some Chroma setups persist automatically; ignore if not supported
        pass

    print(f"Done. Files processed: {files_count}, total chunks indexed: {chunks_count}")


if __name__ == "__main__":
    main(DATA_DIR, CHROMA_DIR, COLLECTION_NAME, BATCH_SIZE)
