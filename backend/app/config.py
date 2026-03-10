from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str

    # Supabase (opcional na fase inicial)
    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_anon_key: str = ""

    # Resend (opcional na fase inicial)
    resend_api_key: str = ""
    email_from: str = "chatbot@instituicao.com"
    email_marketing: str = "marketing@instituicao.com"

    # Agendamento do resumo diário
    daily_report_hour: int = 23
    daily_report_minute: int = 59

    # Caminhos
    pdf_dir: str = "data/pdfs"
    vectorstore_dir: str = "data/vectorstore"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    retriever_k: int = 4

    # Modelos
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
