"""
RAG Chain conversacional:
  - Recupera chunks relevantes do vectorstore para cada mensagem
  - Injeta o contexto + histórico no prompt
  - Gera a resposta via Groq (Llama 3.3 70B)
  - Mantém histórico da conversa em memória (por sessão)
"""
from typing import AsyncIterator

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.agent.prompts import get_chat_prompt
from app.rag.retriever import get_retriever, format_retrieved_docs


def _build_llm(streaming: bool = False) -> ChatGroq:
    return ChatGroq(
        model=settings.llm_model,
        temperature=0.7,
        api_key=settings.groq_api_key,
        streaming=streaming,
    )


class RAGChain:
    """
    Encapsula a chain conversacional RAG para uma única sessão de atendimento.

    Uso:
        chain = RAGChain(vectorstore)
        response = chain.invoke("Quais cursos vocês oferecem?")
    """

    def __init__(self, vectorstore: Chroma):
        self.retriever = get_retriever(vectorstore)
        self.prompt = get_chat_prompt()
        self.llm = _build_llm(streaming=False)
        self.llm_stream = _build_llm(streaming=True)
        self.history: list[HumanMessage | AIMessage] = []

        # Chain síncrona
        self._chain = (
            {
                "context": lambda x: format_retrieved_docs(
                    self.retriever.invoke(x["input"])
                ),
                "history": lambda x: x.get("history", []),
                "input": lambda x: x["input"],
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Chain assíncrona (streaming)
        self._chain_stream = (
            {
                "context": lambda x: format_retrieved_docs(
                    self.retriever.invoke(x["input"])
                ),
                "history": lambda x: x.get("history", []),
                "input": lambda x: x["input"],
            }
            | self.prompt
            | self.llm_stream
            | StrOutputParser()
        )

    def invoke(self, user_message: str) -> str:
        response = self._chain.invoke({
            "input": user_message,
            "history": self.history,
        })
        self._update_history(user_message, response)
        return response

    async def stream(self, user_message: str) -> AsyncIterator[str]:
        full_response = ""
        async for chunk in self._chain_stream.astream({
            "input": user_message,
            "history": self.history,
        }):
            full_response += chunk
            yield chunk
        self._update_history(user_message, full_response)

    def _update_history(self, user_message: str, ai_response: str) -> None:
        self.history.append(HumanMessage(content=user_message))
        self.history.append(AIMessage(content=ai_response))

    def reset(self) -> None:
        self.history = []

    def get_history(self) -> list:
        return self.history

    def history_as_text(self) -> str:
        lines = []
        for msg in self.history:
            role = "Usuário" if isinstance(msg, HumanMessage) else "Agente"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def generate_summary(self) -> str:
        if not self.history:
            return "Conversa sem histórico."

        summary_prompt = (
            "Com base na conversa abaixo entre um potencial aluno e o agente de atendimento, "
            "gere um resumo objetivo em 3 a 5 frases destacando: "
            "perfil do lead, curso de interesse, principais dúvidas/objeções levantadas e próximos passos.\n\n"
            f"{self.history_as_text()}"
        )

        llm = _build_llm()
        response = llm.invoke(summary_prompt)
        return response.content
