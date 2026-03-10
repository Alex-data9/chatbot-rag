"""
Pipeline de ingestão de PDFs:
  1. Carrega todos os PDFs do diretório
  2. Divide em chunks com overlap
  3. Gera embeddings via OpenAI
  4. Persiste no ChromaDB
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from app.config import settings


def load_pdfs(pdf_dir: str) -> list:
    """Carrega todos os PDFs de um diretório, incluindo subpastas."""
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Diretório de PDFs não encontrado: {pdf_dir}")

    pdf_files = list(pdf_path.glob("**/*.pdf"))
    if not pdf_files:
        raise ValueError(f"Nenhum PDF encontrado em: {pdf_dir}")

    loader = DirectoryLoader(
        str(pdf_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()

    # Adiciona metadado de arquivo de origem a cada chunk
    for doc in documents:
        source = Path(doc.metadata.get("source", ""))
        doc.metadata["file_name"] = source.name
        doc.metadata["category"] = _detect_category(source.name)

    print(f"[ingest] {len(pdf_files)} PDFs | {len(documents)} páginas carregadas")
    return documents


def _detect_category(file_name: str) -> str:
    """Infere a categoria do PDF pelo nome do arquivo."""
    name = file_name.lower()
    if "institucional" in name:
        return "institucional"
    if "curso" in name or "catalogo" in name or "catálogo" in name:
        return "cursos"
    if "mercado" in name or "emprego" in name:
        return "mercado"
    if "script" in name or "atendimento" in name:
        return "atendimento"
    return "geral"


def split_documents(documents: list) -> list:
    """Divide os documentos em chunks com sobreposição para melhor contexto."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingest] {len(chunks)} chunks gerados (size={settings.chunk_size}, overlap={settings.chunk_overlap})")
    return chunks


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def create_vectorstore(chunks: list) -> Chroma:
    """Cria e persiste o vectorstore no ChromaDB."""
    Path(settings.vectorstore_dir).mkdir(parents=True, exist_ok=True)

    embeddings = _get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.vectorstore_dir,
        collection_name="chatbot_rag",
    )
    print(f"[ingest] Vectorstore salvo em: {settings.vectorstore_dir}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Carrega um vectorstore já existente do disco."""
    embeddings = _get_embeddings()
    vectorstore = Chroma(
        persist_directory=settings.vectorstore_dir,
        embedding_function=embeddings,
        collection_name="chatbot_rag",
    )
    count = vectorstore._collection.count()
    print(f"[ingest] Vectorstore carregado — {count} chunks indexados")
    return vectorstore


def vectorstore_exists() -> bool:
    """Verifica se já existe um vectorstore persistido."""
    vs_path = Path(settings.vectorstore_dir)
    return vs_path.exists() and any(vs_path.iterdir())


def ingest(pdf_dir: str | None = None, force: bool = False) -> Chroma:
    """
    Executa o pipeline completo de ingestão.

    Args:
        pdf_dir: Caminho para os PDFs. Usa settings.pdf_dir se None.
        force:   Re-ingere mesmo que o vectorstore já exista.
    """
    pdf_dir = pdf_dir or settings.pdf_dir

    if not force and vectorstore_exists():
        print("[ingest] Vectorstore existente encontrado. Use force=True para re-indexar.")
        return load_vectorstore()

    documents = load_pdfs(pdf_dir)
    chunks = split_documents(documents)
    return create_vectorstore(chunks)
