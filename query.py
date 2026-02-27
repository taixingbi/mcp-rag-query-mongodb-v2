# query.py
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from config import settings, get_mongodb_client


# ----------------------------
# LangSmith config
# ----------------------------
def _langsmith_config(
    metadata: Optional[Dict[str, Any]] = None,
    request_id: Optional[Any] = None,
    session_id: Optional[Any] = None,
) -> dict:
    tags = []
    if settings.app_version:
        tags.append(f"app_version:{settings.app_version}")
    if settings.mcp_name:
        tags.append(f"mcp_name:{settings.mcp_name}")
    if request_id is not None:
        tags.append(f"request_id:{request_id}")
    if session_id is not None:
        tags.append(f"session_id:{session_id}")
    out: Dict[str, Any] = {"tags": tags}
    if metadata:
        out["metadata"] = metadata
    return out


# ----------------------------
# Lazy singletons
# ----------------------------
_embedder = None
_mongo_collection = None

# IMPORTANT: chain must be cached per "where" key, not globally
_rag_chain_cache: Dict[str, Any] = {}


def _where_key(where: Optional[Dict[str, Any]]) -> str:
    if not where:
        return "__ALL__"
    try:
        return str(sorted(where.items()))
    except Exception:
        return str(where)


def _default_chunk_k() -> int:
    """Default number of chunks to retrieve (for retriever / ranked output)."""
    return max(settings.retrieval_k * 2, settings.top_k_final)


def _get_embedder() -> OllamaEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
    return _embedder


def _get_mongo_collection():
    """MongoDB collection for dense + BM25 search."""
    global _mongo_collection
    if _mongo_collection is None:
        _mongo_collection = get_mongodb_client()[
            settings.mongodb_db
        ][settings.mongodb_collection]
    return _mongo_collection


def _doc_to_hit(doc: dict, **extra: Any) -> dict:
    """Extract common hit shape from MongoDB document."""
    meta = doc.get("metadata") or {}
    chunk_id = doc.get("chunk_id") or meta.get("chunk_id") or str(doc.get("_id", ""))
    text = doc.get("text") or doc.get("content") or doc.get("page_content", "")
    metadata = meta if meta else {"source": doc.get("source", "")}
    return {"chunk_id": chunk_id, "text": text, "metadata": metadata, **extra}


# ----------------------------
# Dense search (MongoDB Atlas Vector Search)
# ----------------------------
def _search_dense(
    query: str,
    k: int,
    where: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """
    Dense search: embed(query) -> MongoDB Atlas $vectorSearch -> top K.
    """
    coll = _get_mongo_collection()
    q_emb = _get_embedder().embed_query(query)
    num_candidates = max(k * 20, 100)

    vector_stage: Dict[str, Any] = {
        "index": settings.atlas_vector_index,
        "path": settings.atlas_vector_path,
        "queryVector": q_emb,
        "numCandidates": num_candidates,
        "limit": k,
    }
    if where:
        vector_stage["filter"] = where

    pipeline: List[Dict[str, Any]] = [
        {"$vectorSearch": vector_stage},
        {"$addFields": {"vector_score": {"$meta": "vectorSearchScore"}}},
    ]

    out: List[dict] = []
    for doc in coll.aggregate(pipeline):
        score = float(doc.get("vector_score", 0.0))
        dist = 1.0 / (1.0 + max(score, 0.0))  # similarity -> distance (lower=better)
        out.append(_doc_to_hit(doc, distance=dist))
    return out


# ----------------------------
# BM25 recall (Atlas Search)
# ----------------------------
def _search_bm25_atlas(
    query: str,
    k: int,
    where: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """
    BM25 recall via MongoDB Atlas Search. Returns top K docs.
    """
    coll = _get_mongo_collection()
    search_stage: Dict[str, Any] = {
        "text": {"query": query, "path": settings.atlas_search_path},
    }
    if settings.atlas_search_index != "default":
        search_stage["index"] = settings.atlas_search_index
    if where:
        search_stage["filter"] = where

    pipeline: List[Dict[str, Any]] = [
        {"$search": search_stage},
        {"$addFields": {"score": {"$meta": "searchScore"}}},
        {"$limit": k},
    ]

    out: List[dict] = []
    for doc in coll.aggregate(pipeline):
        score = float(doc.get("score", 0.0))
        out.append(_doc_to_hit(doc, search_score=score, distance=0.0))
    return out


# ----------------------------
# Fallback (when dense + BM25 both return empty, e.g. indexes missing)
# ----------------------------
def _search_fallback(k: int, where: Optional[Dict[str, Any]] = None) -> List[dict]:
    """Simple find() fallback when Atlas indexes return nothing."""
    coll = _get_mongo_collection()
    filter_q = where if where else {}
    out: List[dict] = []
    for doc in coll.find(filter_q).limit(k):
        h = _doc_to_hit(doc, distance=0.0, rrf_score=0.0)
        out.append(h)
    return out


# ----------------------------
# RRF fusion
# ----------------------------
def _fuse_rrf(
    dense_hits: List[dict],
    bm25_hits: List[dict],
    k_final: int,
    rrf_k: int = 60,
) -> List[dict]:
    """
    Reciprocal Rank Fusion: merge dense + BM25 lists by chunk_id.
    rrf_score = sum over lists: 1 / (rrf_k + rank)
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, dict] = {}

    def add_list(hits: List[dict]) -> None:
        for rank, h in enumerate(hits, start=1):
            cid = h.get("chunk_id", "") or str((h.get("metadata") or {}).get("chunk_id", ""))
            if not cid:
                continue
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in doc_map:
                doc_map[cid] = dict(h)
            doc_map[cid]["rrf_score"] = rrf_scores[cid]

    add_list(dense_hits)
    add_list(bm25_hits)

    merged = [doc_map[cid] for cid in doc_map]
    merged.sort(key=lambda x: float(x.get("rrf_score", 0.0)), reverse=True)
    return merged[:k_final]


# ----------------------------
# Dual recall + RRF fusion
# ----------------------------
def _search_dual_rrf(
    query: str,
    k: int,
    where: Optional[Dict[str, Any]] = None,
    *,
    top_k_dense: Optional[int] = None,
    top_k_bm25: Optional[int] = None,
    top_k_final: Optional[int] = None,
    rrf_k: Optional[int] = None,
) -> List[dict]:
    """
    Dual recall + RRF:
      1) Dense recall: embed(question) -> MongoDB Atlas Vector Search -> top K_dense
      2) BM25 recall: Atlas Search on text -> top K_bm25
      3) Fuse with RRF, take top K_final
    """
    k_dense = top_k_dense if top_k_dense is not None else settings.top_k_dense
    k_bm25 = top_k_bm25 if top_k_bm25 is not None else settings.top_k_bm25
    k_final = top_k_final if top_k_final is not None else settings.top_k_final
    rrf_val = rrf_k if rrf_k is not None else settings.rrf_k

    # Run dense + BM25 recall in parallel
    with ThreadPoolExecutor(max_workers=2) as ex:
        dense_future = ex.submit(_search_dense, query, k_dense, where)
        bm25_future = ex.submit(_search_bm25_atlas, query, k_bm25, where)
        dense_hits = dense_future.result()
        bm25_hits = bm25_future.result()

    merged = _fuse_rrf(dense_hits, bm25_hits, k_final=k_final, rrf_k=rrf_val)
    used_fallback = False
    if not merged:
        merged = _search_fallback(k=k_final, where=where)
        used_fallback = True
    return merged[:k], used_fallback


# ----------------------------
# Retriever
# ----------------------------
class CloudRetriever(BaseRetriever):
    """
    Retriever using dual recall (dense + BM25) + RRF fusion.
    """
    where: Optional[Dict[str, Any]] = None  # you can set this externally

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        k = _default_chunk_k()
        hits, _ = _search_dual_rrf(query, k=k, where=self.where)
        return [Document(page_content=h["text"], metadata=h["metadata"]) for h in hits]


def get_retriever(where: Optional[Dict[str, Any]] = None) -> CloudRetriever:
    r = CloudRetriever()
    r.where = where
    return r


def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs[:settings.retrieval_k])


# ----------------------------
# Prompt + chain
# ----------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions based only on the provided context. "
            "If the context does not contain relevant information, say so. "
            "Do not make up facts. Cite the context when possible.",
        ),
        ("human", "Context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ]
)


def build_rag_chain(where: Optional[Dict[str, Any]] = None):
    """
    Build RAG chain. Cached per-where (IMPORTANT for correctness if where changes).
    """
    key = _where_key(where)
    if key not in _rag_chain_cache:
        chain = (
            {"context": get_retriever(where=where) | format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | ChatOllama(
                model=settings.ollama_llm_model,
                base_url=settings.ollama_base_url,
                temperature=0,
            )
            | StrOutputParser()
        )
        _rag_chain_cache[key] = chain.with_config(run_name=settings.mcp_name)
    return _rag_chain_cache[key]


def run_query(
    question: str,
    where: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    request_id: Optional[Any] = None,
    session_id: Optional[Any] = None,
) -> str:
    return build_rag_chain(where=where).invoke(
        question,
        config=_langsmith_config(
            metadata=metadata,
            request_id=request_id,
            session_id=session_id,
        ),
    )


# ----------------------------
# Return ranked chunks + answer
# ----------------------------
def retrieve_ranked_chunks(
    question: str,
    k: int | None = None,
    where: Optional[Dict[str, Any]] = None,
    *,
    top_k_dense: Optional[int] = None,
    top_k_bm25: Optional[int] = None,
    top_k_final: Optional[int] = None,
) -> List[dict]:
    """
    Return ranked chunks for debugging / UI.
    Uses dual recall (dense + BM25) + RRF fusion.
    Each chunk has rank (1-based), scores (nested), source, preview, text, metadata.
    """
    k_out = k if k is not None else _default_chunk_k()
    hits, used_fallback = _search_dual_rrf(
        question,
        k=k_out,
        where=where,
        top_k_dense=top_k_dense,
        top_k_bm25=top_k_bm25,
        top_k_final=top_k_final,
    )

    ranked: List[dict] = []
    for rank_1based, h in enumerate(hits, start=1):
        meta = h.get("metadata") or {}
        text = h.get("text") or ""
        chunk_id = h.get("chunk_id", "") or meta.get("chunk_id", "")
        ranked.append(
            {
                "rank": rank_1based,
                "chunk_id": chunk_id,
                "source": meta.get("source", ""),
                "preview": text[:250],
                "text": text,
                "scores": {
                    "rrf_score": h.get("rrf_score", 0.0),
                    "distance": h.get("distance", 0.0),
                    "search_score": h.get("search_score", 0.0),
                },
                "metadata": meta,
            }
        )
    return ranked, used_fallback


def run_query_with_chunks(
    question: str,
    where: Optional[Dict[str, Any]] = None,
    chunk_k: int | None = None,
    request_id: Optional[Any] = None,
    session_id: Optional[Any] = None,
    *,
    top_k_dense: Optional[int] = None,
    top_k_bm25: Optional[int] = None,
    top_k_final: Optional[int] = None,
) -> Dict[str, Any]:
    k_out = chunk_k if chunk_k is not None else _default_chunk_k()
    chunks, used_fallback = retrieve_ranked_chunks(
        question,
        k=k_out,
        where=where,
        top_k_dense=top_k_dense,
        top_k_bm25=top_k_bm25,
        top_k_final=top_k_final,
    )
    answer = run_query(
        question,
        where=where,
        metadata={"reranked_chunks": chunks},
        request_id=request_id,
        session_id=session_id,
    )
    used_k = min(settings.retrieval_k, len(chunks))
    used_chunk_ids = list(dict.fromkeys(c["chunk_id"] for c in chunks[:used_k]))
    warnings: List[str] = []
    if used_fallback:
        warnings.append(
            "fallback_used: dense + BM25 recall returned empty; used simple find(). "
            "Configure Atlas Vector Search and Atlas Search indexes for semantic ranking."
        )
    return {
        "answer": answer,
        "chunks": chunks,
        "used_chunk_ids": used_chunk_ids,
        "retrieval": {
            "k": k_out,
            "top_k_dense": top_k_dense if top_k_dense is not None else settings.top_k_dense,
            "top_k_bm25": top_k_bm25 if top_k_bm25 is not None else settings.top_k_bm25,
            "top_k_final": top_k_final if top_k_final is not None else settings.top_k_final,
            "rrf_k": settings.rrf_k,
            "filters": where or {},
            "warnings": warnings,
        },
        "metadata": {"reranked_chunks": chunks},
    }


if __name__ == "__main__":
    import sys
    import json

    question = input("Question: ").strip() if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    if not question:
        print("Usage: python query.py <question>")
        sys.exit(1)

    result = run_query_with_chunks(question)

    print(f"Q: {question}\n")
    print(f"A: {result['answer']}\n")

    print("Top ranked chunks:\n")
    print(json.dumps(result["chunks"][:settings.retrieval_k], indent=2, ensure_ascii=False))
