"""
Módulo de retrieval: busca os chunks mais relevantes no ChromaDB
para uma dada mensagem do usuário.
"""
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever

from app.config import settings


def get_retriever(vectorstore: Chroma, k: int | None = None) -> BaseRetriever:
    """
    Retorna um retriever de similaridade simples.

    Args:
        vectorstore: Instância do ChromaDB já carregada.
        k:           Número de chunks a recuperar. Usa settings.retriever_k se None.
    """
    k = k or settings.retriever_k
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def get_retriever_with_score(vectorstore: Chroma, k: int | None = None, score_threshold: float = 0.5):
    """
    Retriever com filtro por score mínimo de similaridade.
    Útil para evitar recuperar chunks completamente irrelevantes.
    """
    k = k or settings.retriever_k
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )


def format_retrieved_docs(docs: list) -> str:
    """Formata a lista de documentos recuperados em uma string de contexto."""
    if not docs:
        return "Nenhuma informação relevante encontrada na base de conhecimento."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("file_name", "desconhecido")
        category = doc.metadata.get("category", "geral")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[Fonte {i} — {source} | categoria: {category} | pág. {page}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)
