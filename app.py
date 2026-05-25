"""
Sofia — Chatbot RAG  |  Hugging Face Spaces (Gradio)

Reutiliza toda a lógica Python do backend (rag/, agent/) sem o FastAPI.
Variáveis de ambiente necessárias (configure em Settings > Secrets no HF Space):
  DEEPSEEK_API_KEY   — chave da DeepSeek (LLM)
  OPENAI_API_KEY     — chave da OpenAI (embeddings)
"""
import shutil
import sys
from pathlib import Path

# Aponta o Python para o pacote backend/app
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import gradio as gr

from app.config import settings
from app.rag.ingest import ingest, vectorstore_exists, load_vectorstore
from app.rag.chain import RAGChain

# ── Estado global (demo single-session) ──────────────────────────────────────
_rag_chain: RAGChain | None = None
_rag_empty: bool = True


def _init_rag() -> None:
    global _rag_chain, _rag_empty
    pdf_dir = Path(settings.pdf_dir)
    has_pdfs = pdf_dir.exists() and any(pdf_dir.glob("**/*.pdf"))

    if vectorstore_exists():
        vs = load_vectorstore()
        if vs._collection.count() > 0:
            _rag_chain = RAGChain(vs)
            _rag_empty = False
            return
    if has_pdfs:
        vs = ingest()
        _rag_chain = RAGChain(vs)
        _rag_empty = False


_init_rag()


# ── Funções principais ────────────────────────────────────────────────────────

def _status_md() -> str:
    if _rag_empty:
        return (
            "> ⚠️ **Base de conhecimento vazia.** "
            "Vá até a aba **📂 Carregar PDFs** e envie os documentos da instituição."
        )
    return "> ✅ **Sofia está pronta!** Base de conhecimento carregada."


def respond(message: str, history: list) -> str:
    if _rag_empty or _rag_chain is None:
        return (
            "⚠️ A base de conhecimento ainda está vazia.\n\n"
            "Vá até a aba **📂 Carregar PDFs** e faça o upload dos documentos da instituição."
        )
    return _rag_chain.invoke(message)


def reset_conversation() -> tuple[list, str]:
    if _rag_chain:
        _rag_chain.reset()
    return [], "Conversa reiniciada com sucesso."


def ingest_files(files) -> str:
    global _rag_chain, _rag_empty

    if not files:
        return "Nenhum arquivo selecionado."

    pdf_dir = Path(settings.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        dest = pdf_dir / Path(f.name).name
        shutil.copy(f.name, dest)
        saved.append(dest.name)

    try:
        vs = ingest(force=True)
        _rag_chain = RAGChain(vs)
        _rag_empty = False
        count = vs._collection.count()
        names = "\n".join(f"  • {n}" for n in saved)
        return f"✅ {len(saved)} PDF(s) indexado(s) com sucesso!\n{names}\n\n{count} chunks na base de conhecimento."
    except Exception as exc:
        return f"❌ Erro na ingestão: {exc}"


# ── Interface Gradio ──────────────────────────────────────────────────────────

with gr.Blocks(
    title="Sofia — Chatbot RAG",
    theme=gr.themes.Soft(primary_hue="purple", neutral_hue="slate"),
) as demo:

    gr.Markdown(
        """
        # 🎓 Sofia — Consultora de Matrículas
        *Assistente virtual inteligente com RAG · DeepSeek + LangChain + ChromaDB*
        """
    )

    # ── Aba Chat ──────────────────────────────────────────────────────────────
    with gr.Tab("💬 Chat"):

        status_banner = gr.Markdown(_status_md())

        chatbot = gr.Chatbot(
            label="Sofia",
            height=460,
            bubble_full_width=False,
            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=sofia"),
        )

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Digite sua mensagem… (Enter para enviar, Shift+Enter para nova linha)",
                label="",
                lines=2,
                scale=5,
                show_label=False,
            )
            btn_send = gr.Button("Enviar", variant="primary", scale=1)

        with gr.Row():
            btn_reset = gr.Button("🔄 Nova conversa", variant="secondary")
            reset_status = gr.Textbox(label="", interactive=False, visible=False)

        # Fluxo: usuário envia → resposta do bot
        def _user_turn(message: str, history: list):
            return "", history + [[message, None]]

        def _bot_turn(history: list):
            history[-1][1] = respond(history[-1][0], history[:-1])
            return history

        msg_input.submit(_user_turn, [msg_input, chatbot], [msg_input, chatbot]).then(
            _bot_turn, chatbot, chatbot
        )
        btn_send.click(_user_turn, [msg_input, chatbot], [msg_input, chatbot]).then(
            _bot_turn, chatbot, chatbot
        )
        btn_reset.click(reset_conversation, outputs=[chatbot, reset_status])

    # ── Aba Upload PDFs ───────────────────────────────────────────────────────
    with gr.Tab("📂 Carregar PDFs"):

        gr.Markdown(
            """
            ### Upload da Base de Conhecimento
            Envie os PDFs da instituição (catálogos de cursos, material institucional, scripts de atendimento, etc.).
            Após o upload, clique em **Indexar PDFs** — a Sofia estará pronta em instantes.
            """
        )

        file_input = gr.File(
            file_count="multiple",
            file_types=[".pdf"],
            label="Selecione os PDFs",
        )
        btn_ingest = gr.Button("Indexar PDFs", variant="primary")
        ingest_output = gr.Textbox(label="Status da ingestão", interactive=False, lines=5)

        btn_ingest.click(ingest_files, inputs=file_input, outputs=ingest_output).then(
            lambda: gr.Markdown(_status_md()), outputs=status_banner
        )

    # ── Aba Sobre ─────────────────────────────────────────────────────────────
    with gr.Tab("ℹ️ Sobre"):
        gr.Markdown(
            """
            ## Sofia — Chatbot RAG para Instituições de Ensino

            **Objetivo:** Atender potenciais alunos com tom acolhedor, profissional e entusiasmado,
            guiando-os pelo processo de descoberta do curso ideal e incentivando a matrícula.

            ---

            ### Stack Técnica
            | Componente | Tecnologia |
            |------------|------------|
            | LLM (chat) | DeepSeek Chat |
            | Embeddings | OpenAI text-embedding-3-small |
            | Vectorstore | ChromaDB (local) |
            | RAG framework | LangChain |
            | Interface demo | Gradio / Hugging Face Spaces |
            | Backend prod | FastAPI + uvicorn |

            ---

            ### Fluxo de Atendimento (7 etapas)
            1. **Boas-vindas** — cumprimento e identificação de perfil
            2. **Apresentação do curso** — grade, duração, modalidade
            3. **Mercado de trabalho** — empregabilidade, salário, parceiros
            4. **Estrutura institucional** — infraestrutura, nota MEC, laboratórios
            5. **Financeiro** — mensalidade, FIES, ProUni, bolsas, ROI
            6. **CTA** — coleta de contato, inscrição, visita
            7. **Escalonamento** — transferência para atendimento humano

            ---

            ### Perfis Detectados Automaticamente
            | Perfil | Gatilho |
            |--------|---------|
            | Vestibulando | "vestibular", "ensino médio", "primeiro curso" |
            | Profissional | "já trabalho", "EAD", "noturno", "salário" |
            | Transferido | "transferência", "outra faculdade" |
            | Pós-graduando | "pós-graduação", "MBA", "sou formado" |
            | Pai / Mãe | "meu filho", "minha filha" |

            ---

            ### Configuração no HF Space
            Em **Settings → Variables and secrets**, adicione:
            ```
            DEEPSEEK_API_KEY   = sk-...   (LLM)
            OPENAI_API_KEY     = sk-...   (Embeddings)
            ```
            """
        )

if __name__ == "__main__":
    demo.launch()
