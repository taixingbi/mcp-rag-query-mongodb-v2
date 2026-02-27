# RAG Pipeline

RAG over documents using **MongoDB Atlas** (dense vector search + BM25 text search) and **LangChain**. Dual recall with RRF fusion. Exposes tools via **MCP** (Model Context Protocol) so clients can call `rag_query_with_chunks` over HTTP.

---

## Setup

<!-- Install Python deps in a venv before running or deploying. -->
Create a virtualenv and install dependencies:

```bash
python3.11 -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

<!-- .env is not committed; copy from .env.example or set these keys. -->
Create a `.env` file in the project root. Ensure Ollama is running locally (or set `OLLAMA_BASE_URL` to your Ollama service; for ECS, set this to your Ollama endpoint):

```
APP_VERSION=v:1.01
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
MONGODB_URI=mongodb+srv://...
MONGODB_DB=db_hunt
MONGODB_COLLECTION=collection_taixingbi_dev
ATLAS_VECTOR_INDEX=vector_index
ATLAS_SEARCH_INDEX=default
```

`OLLAMA_EMBEDDING_MODEL` must match the model used to build the vectors in Atlas.

---

## Ollama Service

The RAG pipeline uses Ollama for embeddings (dense search) and for the chat LLM. Run Ollama before starting the MCP server.

**Install Ollama** (see [ollama.com](https://ollama.com)):

```bash
# macOS
brew install ollama
```

**Start the Ollama server** (default: `http://localhost:11434`):

```bash
ollama serve
```

**Pull the models** used in `.env` (LLM and embedding model must match your Atlas index):

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

If you use different `OLLAMA_LLM_MODEL` or `OLLAMA_EMBEDDING_MODEL` in `.env`, pull those instead (e.g. `ollama pull mistral`, `ollama pull mxbai-embed-large`).

---

## Local Development

<!-- All commands below assume you are in the repo root and have activated the venv. -->

### Run MCP HTTP Server

```bash
uvicorn main:app --reload --port 8000
```

### Test Endpoints

**Health check:**

```bash
curl http://127.0.0.1:8000/health
```

**Call MCP tools** (use trailing slash `/mcp/` to avoid 307 redirect):


**`rag_query_with_chunks`** — returns answer + ranked chunks as JSON:

```bash
curl -s --max-time 60 -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": "call-001",
    "method": "tools/call",
    "params": {
      "name": "rag_query_with_chunks",
      "arguments": {
        "question": "what is Taixing visa status?",
        "request_id": "12345678",
        "session_id": "123456"
      }
    }
  }' \
  http://localhost:8000/mcp/
```

Response is JSON-RPC; the tool result is a JSON string with: `answer`, `chunks` (each with `rank`, `chunk_id`, `source`, `preview`, `text`, `scores` (e.g. `rrf_score`, `distance`, `search_score`), `metadata`), `used_chunk_ids` (unique chunk IDs used for the answer), `retrieval` (`k`, `top_k_dense`, `top_k_bm25`, `top_k_final`, `rrf_k`, `filters`, `warnings`).

---

## Docker

<!-- Image does not bundle .env; pass --env-file at run time. -->
Build the image:

```bash
docker build -t rag-mcp .
```

Run in the background (detached). Use `-e OLLAMA_BASE_URL=...` so the container can reach Ollama on the host; omit if your `.env` already sets it or Ollama is at a different URL:

```bash
docker run -d \
  -p 8000:8000 \
  --name rag-mcp-run \
  --env-file .env \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  rag-mcp
```

Then check: `curl http://127.0.0.1:8000/health`. If the name `rag-mcp-run` is already in use, remove it first: `docker rm -f rag-mcp-run`. To stop and remove: `docker stop rag-mcp-run && docker rm rag-mcp-run`. To run again, use the same `docker run` command (remove the existing container first if needed).

**Ollama when running in Docker:** Inside the container, `localhost` is the container, not your machine. If Ollama runs on the host (e.g. your Mac), pass `-e OLLAMA_BASE_URL=http://host.docker.internal:11434` as above, or set it in `.env`. (`host.docker.internal` works on Docker Desktop for Mac and Windows; on Linux you may need `--add-host=host.docker.internal:host-gateway`.) For ECS or another remote Ollama, set `OLLAMA_BASE_URL` to that service URL.