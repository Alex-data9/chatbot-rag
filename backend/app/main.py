"""
FastAPI entry point — Chatbot RAG v2.0

Endpoints:
  GET  /api/health          — Health check
  POST /api/chat            — Resposta síncrona
  POST /api/chat/stream     — Resposta em streaming (SSE)
  POST /api/chat/reset      — Reinicia a conversa
  POST /api/chat/end        — Finaliza e gera resumo
  POST /api/ingest          — Re-indexa os PDFs (admin)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.rag.ingest import ingest, vectorstore_exists, load_vectorstore
from app.rag.chain import RAGChain

# ---------------------------------------------------------------------------
# Estado global (substituir por sessões por usuário em produção)
# ---------------------------------------------------------------------------
_vectorstore = None
_rag_chain: RAGChain | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa o vectorstore e a chain ao subir a aplicação."""
    global _vectorstore, _rag_chain

    if vectorstore_exists():
        print("[main] Carregando vectorstore existente...")
        _vectorstore = load_vectorstore()
    else:
        print("[main] Nenhum vectorstore encontrado. Rodando ingestão...")
        _vectorstore = ingest()

    _rag_chain = RAGChain(_vectorstore)
    print("[main] Agente RAG pronto para atendimento!")
    yield


# ---------------------------------------------------------------------------
# Aplicação
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Chatbot RAG — Instituição de Ensino",
    version="2.0.0",
    description="Agente de atendimento de marketing com RAG sobre base de PDFs.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Restringir em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


class EndConversationResponse(BaseModel):
    summary: str
    session_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "vectorstore_ready": _vectorstore is not None,
        "chain_ready": _rag_chain is not None,
        "model": settings.llm_model,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Envia uma mensagem e recebe a resposta completa."""
    if _rag_chain is None:
        raise HTTPException(status_code=503, detail="Agente RAG não inicializado.")

    response = _rag_chain.invoke(request.message)
    return ChatResponse(response=response, session_id=request.session_id)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Envia uma mensagem e recebe a resposta em streaming (Server-Sent Events)."""
    if _rag_chain is None:
        raise HTTPException(status_code=503, detail="Agente RAG não inicializado.")

    async def generate():
        async for chunk in _rag_chain.stream(request.message):
            # Escapa newlines para SSE
            safe_chunk = chunk.replace("\n", "\\n")
            yield f"data: {safe_chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/chat/reset")
async def reset_chat(session_id: str = "default"):
    """Reinicia o histórico da conversa."""
    if _rag_chain:
        _rag_chain.reset()
    return {"status": "Conversa reiniciada", "session_id": session_id}


@app.post("/api/chat/end", response_model=EndConversationResponse)
async def end_chat(session_id: str = "default"):
    """
    Finaliza a conversa:
      1. Gera resumo via LLM
      2. (Futuro) Salva no Supabase + envia email de relatório
    """
    if _rag_chain is None:
        raise HTTPException(status_code=503, detail="Agente RAG não inicializado.")

    summary = _rag_chain.generate_summary()
    _rag_chain.reset()
    return EndConversationResponse(summary=summary, session_id=session_id)


@app.post("/api/ingest")
async def reingest(force: bool = True):
    """Re-indexa todos os PDFs (uso administrativo)."""
    global _vectorstore, _rag_chain
    _vectorstore = ingest(force=force)
    _rag_chain = RAGChain(_vectorstore)
    return {"status": "Base de conhecimento re-indexada com sucesso."}
