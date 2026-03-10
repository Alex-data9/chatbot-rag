/**
 * Chatbot RAG — Frontend App
 */
(() => {
  "use strict";

  // ── Config ──────────────────────────────────────────────
  const API_BASE = window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost"
    ? `http://${window.location.hostname}:8000`
    : "";

  // ── DOM Elements ────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const chatMessages = $("#chat-messages");
  const chatInput = $("#chat-input");
  const btnSend = $("#btn-send");
  const btnNewChat = $("#btn-new-chat");
  const btnEndChat = $("#btn-end-chat");
  const btnChat = $("#btn-chat");
  const btnStatus = $("#btn-status");
  const viewChat = $("#view-chat");
  const viewStatus = $("#view-status");
  const welcomeScreen = $("#welcome-screen");
  const ragEmptyBanner = $("#rag-empty-banner");
  const connectionStatus = $("#connection-status");
  const statusContent = $("#status-content");

  // ── State ───────────────────────────────────────────────
  let isStreaming = false;
  let ragEmpty = false;
  let hasMessages = false;

  // ── Init ────────────────────────────────────────────────
  checkHealth();
  setupEventListeners();

  // ── Event Listeners ─────────────────────────────────────
  function setupEventListeners() {
    // Send message
    btnSend.addEventListener("click", sendMessage);
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    // Auto-resize textarea
    chatInput.addEventListener("input", () => {
      chatInput.style.height = "auto";
      chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
      btnSend.disabled = !chatInput.value.trim() || isStreaming;
    });

    // Quick actions
    $$(".quick-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        chatInput.value = btn.dataset.msg;
        chatInput.dispatchEvent(new Event("input"));
        sendMessage();
      });
    });

    // Navigation
    btnChat.addEventListener("click", () => switchView("chat"));
    btnStatus.addEventListener("click", () => {
      switchView("status");
      loadStatus();
    });

    // New chat
    btnNewChat.addEventListener("click", resetChat);
    btnEndChat.addEventListener("click", endChat);
  }

  // ── View Switching ──────────────────────────────────────
  function switchView(view) {
    $$(".view").forEach((v) => v.classList.remove("active"));
    $$(".nav-btn").forEach((b) => b.classList.remove("active"));

    if (view === "chat") {
      viewChat.classList.add("active");
      btnChat.classList.add("active");
    } else {
      viewStatus.classList.add("active");
      btnStatus.classList.add("active");
    }
  }

  // ── Health Check ────────────────────────────────────────
  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      const data = await res.json();

      ragEmpty = data.rag_empty === true;

      if (ragEmpty) {
        connectionStatus.className = "status-indicator empty";
        connectionStatus.querySelector(".status-text").textContent = "RAG vazio";
        ragEmptyBanner.classList.remove("hidden");
      } else {
        connectionStatus.className = "status-indicator online";
        connectionStatus.querySelector(".status-text").textContent = "Online";
        ragEmptyBanner.classList.add("hidden");
      }
    } catch {
      connectionStatus.className = "status-indicator offline";
      connectionStatus.querySelector(".status-text").textContent = "Offline";
    }
  }

  // ── Send Message ────────────────────────────────────────
  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || isStreaming) return;

    // Hide welcome
    if (!hasMessages) {
      welcomeScreen.classList.add("hidden");
      hasMessages = true;
    }

    // Add user message
    appendMessage("user", text);
    chatInput.value = "";
    chatInput.style.height = "auto";
    btnSend.disabled = true;
    isStreaming = true;

    // Show typing
    const typingEl = showTyping();

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      const data = await res.json();
      removeTyping(typingEl);

      if (data.rag_empty || data.response === null) {
        ragEmpty = true;
        ragEmptyBanner.classList.remove("hidden");
        appendMessage(
          "assistant",
          "O RAG está vazio — nenhum documento foi carregado na base de conhecimento.\n\n" +
          "Resultado: none\n\n" +
          "Para utilizar o chatbot, carregue PDFs no diretório data/pdfs/ e execute a ingestão via POST /api/ingest."
        );
      } else {
        appendMessage("assistant", data.response);
      }
    } catch (err) {
      removeTyping(typingEl);
      appendMessage(
        "assistant",
        "Erro ao conectar com o servidor. Verifique se o backend está rodando."
      );
    }

    isStreaming = false;
    btnSend.disabled = !chatInput.value.trim();
  }

  // ── Append Message ──────────────────────────────────────
  function appendMessage(role, text) {
    const div = document.createElement("div");
    div.className = `message ${role}`;

    const avatarText = role === "assistant" ? "S" : "U";

    div.innerHTML = `
      <div class="message-avatar">${avatarText}</div>
      <div class="message-content">${formatText(text)}</div>
    `;

    chatMessages.appendChild(div);
    scrollToBottom();
  }

  function formatText(text) {
    // Basic formatting: newlines to <br>, bold (**text**)
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br>");
  }

  // ── Typing Indicator ───────────────────────────────────
  function showTyping() {
    const div = document.createElement("div");
    div.className = "message assistant";
    div.id = "typing-msg";
    div.innerHTML = `
      <div class="message-avatar">S</div>
      <div class="message-content">
        <div class="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
    return div;
  }

  function removeTyping(el) {
    if (el && el.parentNode) el.parentNode.removeChild(el);
  }

  // ── Scroll ──────────────────────────────────────────────
  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // ── Reset Chat ──────────────────────────────────────────
  async function resetChat() {
    try {
      await fetch(`${API_BASE}/api/chat/reset`, { method: "POST" });
    } catch { /* ignore */ }

    // Clear UI
    chatMessages.innerHTML = "";
    chatMessages.appendChild(welcomeScreen);
    welcomeScreen.classList.remove("hidden");
    hasMessages = false;
  }

  // ── End Chat ────────────────────────────────────────────
  async function endChat() {
    if (!hasMessages) return;

    const typingEl = showTyping();

    try {
      const res = await fetch(`${API_BASE}/api/chat/end`, { method: "POST" });
      const data = await res.json();
      removeTyping(typingEl);

      appendMessage(
        "assistant",
        `**Conversa encerrada.**\n\n**Resumo:**\n${data.summary}`
      );
    } catch {
      removeTyping(typingEl);
      appendMessage("assistant", "Erro ao encerrar a conversa.");
    }
  }

  // ── Load Status ─────────────────────────────────────────
  async function loadStatus() {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      const data = await res.json();

      statusContent.innerHTML = `
        <div class="status-card">
          <h3>Sistema</h3>
          <div class="status-row">
            <span class="status-label">Status</span>
            <span class="status-value ok">${data.status}</span>
          </div>
          <div class="status-row">
            <span class="status-label">Modelo LLM</span>
            <span class="status-value">${data.model}</span>
          </div>
          <div class="status-row">
            <span class="status-label">Vectorstore</span>
            <span class="status-value ${data.vectorstore_ready ? 'ok' : 'err'}">
              ${data.vectorstore_ready ? 'Carregado' : 'Indisponível'}
            </span>
          </div>
          <div class="status-row">
            <span class="status-label">Chain RAG</span>
            <span class="status-value ${data.chain_ready ? 'ok' : 'err'}">
              ${data.chain_ready ? 'Ativa' : 'Inativa'}
            </span>
          </div>
          <div class="status-row">
            <span class="status-label">Base de Conhecimento</span>
            <span class="status-value ${data.rag_empty ? 'warn' : 'ok'}">
              ${data.rag_empty ? 'Vazia (nenhum documento)' : 'Populada'}
            </span>
          </div>
        </div>

        <div class="status-card">
          <h3>Endpoints Disponíveis</h3>
          <div class="status-row">
            <span class="status-label">GET /api/health</span>
            <span class="status-value">Health check</span>
          </div>
          <div class="status-row">
            <span class="status-label">POST /api/chat</span>
            <span class="status-value">Chat síncrono</span>
          </div>
          <div class="status-row">
            <span class="status-label">POST /api/chat/stream</span>
            <span class="status-value">Chat com streaming</span>
          </div>
          <div class="status-row">
            <span class="status-label">POST /api/chat/reset</span>
            <span class="status-value">Reiniciar conversa</span>
          </div>
          <div class="status-row">
            <span class="status-label">POST /api/chat/end</span>
            <span class="status-value">Encerrar e resumir</span>
          </div>
          <div class="status-row">
            <span class="status-label">POST /api/ingest</span>
            <span class="status-value">Re-indexar PDFs</span>
          </div>
        </div>
      `;
    } catch {
      statusContent.innerHTML = `
        <div class="status-card">
          <h3>Erro</h3>
          <p style="color: var(--error);">Não foi possível conectar ao backend.</p>
        </div>
      `;
    }
  }
})();
