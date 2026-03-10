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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

from app.config import settings
from app.rag.ingest import ingest, vectorstore_exists, load_vectorstore
from app.rag.chain import RAGChain

# ---------------------------------------------------------------------------
# Estado global (substituir por sessões por usuário em produção)
# ---------------------------------------------------------------------------
_vectorstore = None
_rag_chain: RAGChain | None = None
_rag_empty: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa o vectorstore e a chain ao subir a aplicação."""
    global _vectorstore, _rag_chain, _rag_empty

    pdf_dir = Path(settings.pdf_dir)
    has_pdfs = pdf_dir.exists() and any(pdf_dir.glob("**/*.pdf"))

    if vectorstore_exists():
        print("[main] Carregando vectorstore existente...")
        _vectorstore = load_vectorstore()
        count = _vectorstore._collection.count()
        _rag_empty = count == 0
        if not _rag_empty:
            _rag_chain = RAGChain(_vectorstore)
    elif has_pdfs:
        print("[main] Nenhum vectorstore encontrado. Rodando ingestão...")
        _vectorstore = ingest()
        _rag_empty = False
        _rag_chain = RAGChain(_vectorstore)
    else:
        print("[main] RAG vazio: nenhum PDF encontrado e nenhum vectorstore existente.")
        _rag_empty = True

    if _rag_empty:
        print("[main] O RAG está vazio. O chatbot responderá que não há dados disponíveis.")
    else:
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

# Servir o frontend estático
frontend_path = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str | None
    session_id: str
    rag_empty: bool = False


class EndConversationResponse(BaseModel):
    summary: str
    session_id: str


# ---------------------------------------------------------------------------
# Mensagem padrão quando RAG está vazio
# ---------------------------------------------------------------------------
RAG_EMPTY_MESSAGE = (
    "O sistema RAG está vazio. Nenhum documento foi carregado na base de conhecimento. "
    "Por favor, faça o upload de PDFs no diretório 'data/pdfs/' e execute a ingestão "
    "através do endpoint POST /api/ingest."
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "vectorstore_ready": _vectorstore is not None,
        "chain_ready": _rag_chain is not None,
        "rag_empty": _rag_empty,
        "model": settings.llm_model,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Envia uma mensagem e recebe a resposta completa."""
    if _rag_empty or _rag_chain is None:
        return ChatResponse(
            response=None,
            session_id=request.session_id,
            rag_empty=True,
        )

    response = _rag_chain.invoke(request.message)
    return ChatResponse(response=response, session_id=request.session_id, rag_empty=False)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Envia uma mensagem e recebe a resposta em streaming (Server-Sent Events)."""
    if _rag_empty or _rag_chain is None:
        async def empty_response():
            yield f"data: {RAG_EMPTY_MESSAGE}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty_response(), media_type="text/event-stream")

    async def generate():
        async for chunk in _rag_chain.stream(request.message):
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
    if _rag_empty or _rag_chain is None:
        return EndConversationResponse(
            summary="RAG vazio — nenhuma conversa registrada.",
            session_id=session_id,
        )

    summary = _rag_chain.generate_summary()
    _rag_chain.reset()
    return EndConversationResponse(summary=summary, session_id=session_id)


@app.post("/api/ingest")
async def reingest(force: bool = True):
    """Re-indexa todos os PDFs (uso administrativo)."""
    global _vectorstore, _rag_chain, _rag_empty
    try:
        _vectorstore = ingest(force=force)
        _rag_chain = RAGChain(_vectorstore)
        _rag_empty = False
        return {"status": "Base de conhecimento re-indexada com sucesso."}
    except (FileNotFoundError, ValueError) as e:
        _rag_empty = True
        return {"status": f"Erro na ingestão: {e}", "rag_empty": True}
