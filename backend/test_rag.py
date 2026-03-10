"""
Script de teste CLI — use para validar o RAG sem precisar subir a API.

Uso:
  1. Copie .env.example para .env e preencha OPENAI_API_KEY
  2. Coloque seus PDFs em data/pdfs/
  3. Execute: python test_rag.py

Flags:
  --ingest   Força re-indexação dos PDFs antes de iniciar o chat
  --pdf-dir  Caminho alternativo para os PDFs (default: data/pdfs)
"""
import argparse
import sys
import os

# Garante que o diretório do backend está no PYTHONPATH
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from app.rag.ingest import ingest, vectorstore_exists, load_vectorstore
from app.rag.chain import RAGChain


SEPARATOR = "─" * 60


def print_header():
    print(f"\n{'═' * 60}")
    print("  Chatbot RAG — Instituição de Ensino  |  Teste CLI")
    print(f"{'═' * 60}")
    print("  Digite sua mensagem e pressione Enter.")
    print("  Comandos especiais:")
    print("    /sair    — encerra o chat")
    print("    /reset   — reinicia o histórico")
    print("    /resumo  — gera resumo da conversa")
    print(f"{'═' * 60}\n")


def run_chat(chain: RAGChain):
    print_header()
    print("Agente: Olá! Sou a Sofia, consultora de matrículas. "
          "Como posso te ajudar hoje?\n")

    while True:
        try:
            user_input = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nEncerrando...")
            break

        if not user_input:
            continue

        # Comandos especiais
        if user_input.lower() in ("/sair", "/exit", "/quit"):
            print("\nAgente: Foi um prazer! Até logo. 😊")
            break

        if user_input.lower() == "/reset":
            chain.reset()
            print("[Sistema] Histórico reiniciado.\n")
            continue

        if user_input.lower() == "/resumo":
            print(f"\n{SEPARATOR}")
            print("RESUMO DA CONVERSA:")
            print(SEPARATOR)
            print(chain.generate_summary())
            print(f"{SEPARATOR}\n")
            continue

        # Resposta do agente
        print(f"\n{SEPARATOR}")
        print("Agente: ", end="", flush=True)
        response = chain.invoke(user_input)
        print(response)
        print(f"{SEPARATOR}\n")


def main():
    parser = argparse.ArgumentParser(description="Teste CLI do Chatbot RAG")
    parser.add_argument("--ingest", action="store_true",
                        help="Força re-indexação dos PDFs")
    parser.add_argument("--pdf-dir", default=None,
                        help="Caminho para os PDFs (default: data/pdfs)")
    args = parser.parse_args()

    # Carrega ou cria o vectorstore
    if args.ingest or not vectorstore_exists():
        if not vectorstore_exists():
            print("[setup] Nenhum vectorstore encontrado. Iniciando ingestão...")
        else:
            print("[setup] Forçando re-indexação dos PDFs...")
        vectorstore = ingest(pdf_dir=args.pdf_dir, force=True)
    else:
        print("[setup] Carregando vectorstore existente...")
        vectorstore = load_vectorstore()

    chain = RAGChain(vectorstore)
    run_chat(chain)


if __name__ == "__main__":
    main()
