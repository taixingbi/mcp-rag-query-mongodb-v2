# main.py — MCP HTTP server exposing RAG tools
import contextlib

from fastapi import FastAPI
from mcp.server import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from config import settings
from query import run_query_with_chunks

# streamable_http_path="/" so mounted at /mcp matches (path becomes /)
mcp = FastMCP(
    settings.mcp_name,
    stateless_http=True,
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    ),
)

@mcp.tool()
def rag_query_with_chunks(
    question: str,
    request_id: int | str | None = None,
    session_id: int | str | None = None,
    where: dict | None = None,
    top_k_dense: int | None = None,
    top_k_bm25: int | None = None,
    top_k_final: int | None = None,
):
    """Answer plus top ranked chunks. Uses dual recall (dense + BM25) + RRF fusion.
    Optional: where (tags/doc_id filter), top_k_dense, top_k_bm25, top_k_final.
    request_id and session_id are sent as LangSmith tags."""
    result = run_query_with_chunks(
        question,
        where=where,
        request_id=request_id,
        session_id=session_id,
        top_k_dense=top_k_dense,
        top_k_bm25=top_k_bm25,
        top_k_final=top_k_final,
    )
    return {
        "metadata": result["metadata"],
        "error": None,
        "data": {
            "chunks": result["chunks"],
            "used_chunk_ids": result["used_chunk_ids"],
            "retrieval": result["retrieval"],
            "question": question,
            "answer": result["answer"],
        },
    }


mcp_app = mcp.streamable_http_app()


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    async with mcp.session_manager.run():
        yield


app = FastAPI(title=settings.mcp_name, version="0.1.0", lifespan=_lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mcp": settings.mcp_name,
        "version": settings.app_version,
        "LANGCHAIN_PROJECT": settings.langchain_project,
        "MONGODB_DB": settings.mongodb_db,
        "MONGODB_COLLECTION": settings.mongodb_collection,
    }

app.mount("/mcp", mcp_app)