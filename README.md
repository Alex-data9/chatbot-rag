---
title: Sofia Chatbot RAG
emoji: 🎓
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.0.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🤖 Sofia — Chatbot RAG com Groq + LangChain + FastAPI

Um assistente virtual inteligente para atendimento de marketing em instituições de ensino, alimentado por RAG (Retrieval-Augmented Generation) com DeepSeek API, LangChain, ChromaDB e FastAPI.

**Status:** ✅ Funcional | RAG vazio (aguardando PDFs)
**Modelo:** DeepSeek Chat (deepseek-chat)
**Framework:** FastAPI + LangChain + ChromaDB
**Frontend:** React-like vanilla JS com Dark Theme

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- DeepSeek API Key (https://platform.deepseek.com)

### Instalação & Execução

#### 1. Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt

# Criar arquivo .env
echo 'DEEPSEEK_API_KEY=sk-seu-token-aqui' > .env

# Rodar servidor
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### 2. Frontend (Standalone)
```bash
cd frontend
python -m http.server 8080
```

#### 3. Acessar
- **Interface Web:** http://localhost:8080
- **API Docs:** http://localhost:8000/docs (Swagger UI)
- **Health Check:** http://localhost:8000/api/health

---

## 📋 Estrutura do Projeto

```
chatbot-rag/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI entry point
│   │   ├── config.py               # Settings via Pydantic
│   │   ├── agent/
│   │   │   ├── prompts.py          # System prompt "Sofia"
│   │   │   └── profiles.py         # Lead profile detection
│   │   └── rag/
│   │       ├── chain.py            # RAG conversational chain
│   │       ├── ingest.py           # PDF ingestion pipeline
│   │       └── retriever.py        # ChromaDB retriever
│   ├── .env                        # API Keys (git-ignored)
│   ├── .env.example                # Template
│   ├── requirements.txt            # Python dependencies
│   └── test_rag.py                 # CLI tester
│
├── frontend/
│   ├── index.html                  # Main HTML
│   ├── style.css                   # Dark theme styling
│   ├── app.js                      # Client-side logic
│   └── (assets served as-is)
│
└── README.md                        # Este arquivo
```

---

## 🔧 Configuração

### Backend (.env)
```env
# DeepSeek API
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Supabase (optional, future use)
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
SUPABASE_ANON_KEY=

# Resend Email (optional)
RESEND_API_KEY=
EMAIL_FROM=chatbot@instituicao.com
EMAIL_MARKETING=marketing@instituicao.com

# Scheduling
DAILY_REPORT_HOUR=23
DAILY_REPORT_MINUTE=59
```

### Variáveis de Configuração (appsettings)
Editáveis em `backend/app/config.py`:

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `llm_model` | `deepseek-chat` | Modelo LLM |
| `embedding_model` | `text-embedding-3-small` | Embedding model (OpenAI) |
| `pdf_dir` | `data/pdfs` | Diretório de PDFs |
| `vectorstore_dir` | `data/vectorstore` | Persistência ChromaDB |
| `chunk_size` | `1000` | Tamanho dos chunks |
| `chunk_overlap` | `200` | Overlap entre chunks |
| `retriever_k` | `4` | Top-k docs para retrieval |

---

## 📡 API Endpoints

### Health Check
```bash
GET /api/health
```
**Response (RAG vazio):**
```json
{
  "status": "ok",
  "vectorstore_ready": false,
  "chain_ready": false,
  "rag_empty": true,
  "model": "deepseek-chat"
}
```

### Chat Síncrono
```bash
POST /api/chat
Content-Type: application/json

{
  "message": "Quais cursos vocês oferecem?",
  "session_id": "default"
}
```

**Response (RAG vazio):**
```json
{
  "response": null,
  "session_id": "default",
  "rag_empty": true
}
```

**Response (RAG populado):**
```json
{
  "response": "Olá! Somos uma instituição de ensino...",
  "session_id": "default",
  "rag_empty": false
}
```

### Chat com Streaming (SSE)
```bash
POST /api/chat/stream
Content-Type: application/json

{
  "message": "Quais cursos vocês oferecem?",
  "session_id": "default"
}
```

Retorna tokens via Server-Sent Events em tempo real.

### Reset Conversa
```bash
POST /api/chat/reset?session_id=default
```

Limpa histórico da sessão.

### Encerrar Conversa + Resumo
```bash
POST /api/chat/end?session_id=default
```

**Response:**
```json
{
  "summary": "Lead interessado em cursos de TI...",
  "session_id": "default"
}
```

### Re-indexar PDFs (Admin)
```bash
POST /api/ingest?force=true
```

**Workflow:**
1. Coloca PDFs em `backend/data/pdfs/`
2. Chama `POST /api/ingest`
3. Backend carrega, divide em chunks, gera embeddings, persiste no ChromaDB

---

## 🧠 Lógica RAG

### Pipeline de Ingestão
```
PDFs (data/pdfs/)
    ↓
DirectoryLoader + PyPDFLoader
    ↓
Documentos com metadados (filename, category, page)
    ↓
RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
    ↓
Chunks
    ↓
OpenAIEmbeddings (text-embedding-3-small)
    ↓
ChromaDB (persistent em data/vectorstore/)
```

### Fluxo de Chat
```
User Message
    ↓
Retriever: similarity search (top-4 chunks)
    ↓
Format Context: [Source | Category | Page | Content]
    ↓
Prompt Template:
  - System: Sofia persona + 7-step flow
  - History: últimas mensagens
  - Context: chunks recuperados
  - Input: mensagem do usuário
    ↓
DeepSeek LLM (temperature=0.7)
    ↓
StrOutputParser
    ↓
Response + History Update
```

---

## 👤 Sistema de Perfis de Lead

Detecta automaticamente o tipo de aluno:

| Perfil | Trigger | Foco |
|--------|---------|------|
| **VESTIBULANDO** | "primeiro curso", "saída do ensino médio" | Primeiro emprego, estágio, campus life |
| **PROFISSIONAL** | "já trabalho", "carreira" | Flexibilidade, EAD, crescimento salarial |
| **TRANSFERIDO** | "transferência", "outra faculdade" | Aproveitamento de disciplinas |
| **PÓS-GRADUANDO** | "especialização", "master" | Aprofundamento, networking |
| **PAI/MÃE** | "meu filho", "minha filha" | Segurança, empregabilidade, financiamento |

---

## 💬 Sofia — System Prompt

**Propósito:** Consultora de matrículas com tom ACOLHEDOR, PROFISSIONAL e ENTUSIASMADO.

**7 Etapas de Atendimento:**
1. **Boas-vindas** — Cumprimento + identificação de perfil
2. **Apresentação do Curso** — Destáques, grade, modalidade
3. **Mercado de Trabalho** — Empregabilidade, salário, parceiros
4. **Estrutura Institucional** — Infraestrutura, MEC, acreditações
5. **Financeiro** — Mensalidade, FIES, ProUni, bolsas, ROI
6. **Call-to-Action** — Inscrição, visita, consultor, email/telefone
7. **Escalation** — Transfer para humano se necessário

**Contorno de Objeções:**
- "É muito caro" → FIES, ProUni, bolsas, ROI
- "Não tenho tempo" → EAD, híbrido, noturno, flexibilidade
- "Não sei se é para mim" → Perfil quiz, cursos alternativos
- "Vou pensar" → Urgência real (prazo inscrição, vagas)
- "Já pesquisei outros" → Comparação direta, diferenciais

**Regras Obrigatórias:**
- ✅ Sempre português do Brasil
- ✅ Coletar nome + email OU telefone
- ✅ Basear TUDO em contexto do knowledge base
- ✅ Max 3-4 parágrafos por resposta
- ❌ NUNCA inventar dados/preços/notas
- ❌ NUNCA discussões fora do escopo educacional

---

## 🎨 Frontend Features

### Componentes
- **Sidebar** com navegação (Chat, Status)
- **Chat Panel** com histórico scrollable
- **Welcome Screen** com quick actions
- **Status Page** com health check em tempo real
- **Input Zone** com suporte a markdown
- **RAG Empty Banner** com aviso amarelo

### Atalhos de Teclado
- `Enter` — Enviar mensagem
- `Shift+Enter` — Nova linha
- Textarea auto-redimensiona

### Dark Theme
- Background: `#0f0f14` (off-black)
- Accent: `#6c5ce7` (purple)
- Text: `#e8e8ed` (light gray)
- Gradients suaves nas CTAs

---

## 🔐 Segurança

### Produção
- [ ] Restringir CORS a domínios específicos
- [ ] Usar HTTPS + TLS
- [ ] Implementar rate limiting
- [ ] Validar e sanitizar entrada JSON
- [ ] Usar per-user sessions (não global chain)
- [ ] Armazenar embeddings + histórico no Supabase
- [ ] JWT tokens para admin endpoints

### Atual (Dev)
- ✅ CORS habilitado globalmente (dev convenience)
- ✅ Global chain (single-session)

---

## 📊 Estrutura de Dados

### Document Metadata
```python
{
  "source": "/path/to/file.pdf",
  "file_name": "file.pdf",
  "category": "cursos|institucional|mercado|atendimento|geral",
  "page": 1
}
```

### Message History
```python
[
  HumanMessage(content="Olá!"),
  AIMessage(content="Oi! Bem-vindo..."),
  HumanMessage(content="Quais cursos?"),
  AIMessage(content="Ofereçemos..."),
]
```

### ChromaDB Collection
- **Name:** `chatbot_rag`
- **Embeddings:** OpenAI text-embedding-3-small
- **Persistence:** `backend/data/vectorstore/`
- **Documents:** Auto-indexed com metadata

---

## 🛠️ Desenvolvimento

### Instalar Dependências para Dev
```bash
cd backend
pip install -r requirements.txt
pip install pytest jupyter  # optional
```

### Rodar Testes
```bash
python test_rag.py
# ou use a CLI interativa
```

### Modo Debug
```bash
DEEPSEEK_API_KEY=sk-... python -m uvicorn app.main:app --reload --log-level debug
```

### Adicionar Nova Rota
```python
@app.post("/api/custom")
async def custom_endpoint(request: SomeModel):
    # logic
    return {"result": "..."}
```

---

## 🚀 Deploy

### Docker (Future)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/app ./app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Vercel/Fly.io (Future)
- Frontend: Static files → Vercel
- Backend: FastAPI → Fly.io ou Railway

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: langchain.text_splitter"
**Solução:**
```python
# ❌ Errado
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Correto
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

### RAG vazio / Chain não inicializa
**Verifique:**
1. Há PDFs em `backend/data/pdfs/`?
2. Execute `POST /api/ingest?force=true`
3. Verifique `backend/data/vectorstore/` foi criado

### CORS errors (Frontend)
**Dev workaround:** O backend já tem `allow_origins=["*"]`
**Produção:** Ajuste em `app/main.py` line 76

### DeepSeek API Key inválida
```bash
# Teste a conexão
curl -X POST https://api.deepseek.com/chat/completions \
  -H "Authorization: Bearer sk-seu-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Teste"}]
  }'
```

---

## 📅 Roadmap

- [ ] Integração Supabase (persistir conversas)
- [ ] Email reports via Resend
- [ ] APScheduler para relatórios diários
- [ ] Score-threshold retriever (evitar irrelevantes)
- [ ] Per-user sessions (não global)
- [ ] WebSocket support (em vez de SSE)
- [ ] Admin dashboard (ingestão visual)
- [ ] Analytics (conversão, tópicos)
- [ ] Multi-language support
- [ ] Integração WhatsApp/Telegram

---

## 📝 Changelog

### v2.1.0 (Current)
- ✅ Frontend estático (vanilla JS)
- ✅ Suporte RAG vazio (retorna null + rag_empty flag)
- ✅ Migração DeepSeek (OpenAI → DeepSeek API)
- ✅ Dark theme moderno
- ✅ Health check + status page

### v2.0.0 (Initial)
- FastAPI + LangChain + ChromaDB
- Sofia persona + 7-step flow
- PDF ingestion pipeline
- Session management

---

## 🤝 Contribuindo

1. Clone o repo
2. Crie uma branch (`git checkout -b feature/nome`)
3. Commit (`git commit -am 'Add feature'`)
4. Push (`git push origin feature/nome`)
5. Abra um Pull Request

---

## 📄 Licença

MIT License — Veja [LICENSE](LICENSE) para detalhes.

---

## 👥 Autores

- **Claude Code** — AI Assistant
- **Alex Data** — Projeto Original

---

## 💬 Suporte

Para questões ou bugs:
1. Abra uma [Issue no GitHub](https://github.com/Alex-data9/chatbot-rag/issues)
2. Verifique os troubleshooting acima
3. Consulte a documentação do LangChain/FastAPI

---

## 🎯 Status do Projeto

| Componente | Status | Notas |
|------------|--------|-------|
| Backend API | ✅ Live | Porta 8000 |
| Frontend | ✅ Live | Porta 8080 |
| RAG | ⚠️ Vazio | Aguardando PDFs |
| DeepSeek Integration | ✅ Pronto | API key no .env |
| Supabase | 🔄 Planejado | Fase 2 |
| Email Reports | 🔄 Planejado | Fase 2 |
| Per-user Sessions | 🔄 Planejado | Fase 2 |

---

**🚀 Pronto para usar! Coloque seus PDFs em `backend/data/pdfs/` e boa sorte!**
