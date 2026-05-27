/**
 * ML INTERN — AI Pipeline Agent Frontend
 * Purple Nebula theme · File upload · Chat · Artifact downloads
 */
(function(){
"use strict";

/* ── State ── */
const st = {
  sessions: [],      // [{id, filename, rows, columns}]
  activeId: null,     // current session_id (int)
  messages: [],       // [{role, content, created_at}]
  files: [],          // [{name, size, type}]
  loading: false,
};

const $ = s => document.querySelector(s);
const el = {
  sidebar: $("#sidebar"), overlay: $("#sidebar-overlay"), btnMenu: $("#btn-menu"),
  btnNewSession: $("#btn-new-session"), fileInput: $("#file-input"), btnAttach: $("#btn-attach"), sList: $("#session-list"),
  welcome: $("#welcome"), chatArea: $("#chat-area"), msgs: $("#messages"), typing: $("#typing"),
  input: $("#msg-input"), btnSend: $("#btn-send"),
  filesPanel: $("#files-panel"), filesList: $("#files-list"), btnToggleFiles: $("#btn-toggle-files"),
};

/* ── File Upload & New Session ── */
function setupUpload() {
  el.btnNewSession.addEventListener("click", async () => {
    try {
      el.btnNewSession.classList.add("uploading");
      const resp = await fetch("/session", { method: "POST" });
      const data = await resp.json();
      st.activeId = data.session_id;
      st.messages = [];
      st.files = [];
      const session = {
        id: data.session_id,
        filename: "New Session",
        rows: "-",
        columns: "-",
      };
      st.sessions.unshift(session);
      renderSessions();
      renderMessages();
      closeSidebar();
      el.input.disabled = false;
      el.input.placeholder = "Ask ML Intern anything...";
      st.messages.push({
        role: "assistant",
        content: `Session started (ID: ${data.session_id}). Attach CSV files to begin.`,
        created_at: new Date().toISOString(),
      });
      renderMessages();
    } catch (err) {
      console.error(err);
      alert("Failed to create session");
    } finally {
      el.btnNewSession.classList.remove("uploading");
    }
  });

  el.fileInput.addEventListener("change", async (e) => {
    const files = e.target.files;
    if (!files.length) return;

    if (!st.activeId) {
      try {
        const resp = await fetch("/session", { method: "POST" });
        const data = await resp.json();
        st.activeId = data.session_id;
        st.sessions.unshift({ id: data.session_id, filename: "New Session", rows: "-", columns: "-" });
        el.input.disabled = false;
        el.input.placeholder = "Ask ML Intern anything...";
      } catch (err) {
        alert("Failed to create session");
        return;
      }
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      if (!files[i].name.endsWith(".csv")) {
        alert("Only CSV files are allowed.");
        return;
      }
      formData.append("files", files[i]);
    }

    if (el.btnAttach) el.btnAttach.classList.add("uploading");

    try {
      const resp = await fetch(`/session/${st.activeId}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Upload failed (${resp.status})`);
      }

      const data = await resp.json();
      
      const sess = st.sessions.find(s => s.id === st.activeId);
      if (sess && data.filenames && data.filenames.length) {
          sess.filename = data.filenames[0] + (data.filenames.length > 1 ? ` (+${data.filenames.length-1})` : "");
          renderSessions();
      }

      st.messages.push({
        role: "assistant",
        content: `📂 **${data.files_added} file(s)** loaded successfully: ${data.filenames.join(", ")}.\n\n` +
          `I'm ready to help you build an ML pipeline. What would you like to do first?`,
        created_at: new Date().toISOString(),
      });
      renderMessages();

    } catch (err) {
      console.error("Upload failed:", err);
      alert("Upload failed: " + err.message);
    } finally {
      if (el.btnAttach) el.btnAttach.classList.remove("uploading");
      el.fileInput.value = "";
    }
  });
}

/* ── Chat ── */
async function send() {
  const txt = el.input.value.trim();
  if (!txt || st.loading || !st.activeId) return;

  // Add user message
  st.messages.push({ role: "user", content: txt, created_at: new Date().toISOString() });
  el.input.value = "";
  el.input.style.height = "auto";
  el.btnSend.disabled = true;
  renderMessages();
  st.loading = true;
  showTyping(true);

  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: st.activeId, message: txt }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `Chat failed (${resp.status})`);
    }

    const data = await resp.json();

    // Add AI response
    st.messages.push({
      role: "assistant",
      content: data.response,
      created_at: new Date().toISOString(),
    });

    // Update files
    if (data.files && data.files.length > 0) {
      await refreshFiles();
    }

  } catch (err) {
    console.error("Chat error:", err);
    st.messages.push({
      role: "assistant",
      content: `⚠️ ${err.message}`,
      created_at: new Date().toISOString(),
    });
  } finally {
    st.loading = false;
    showTyping(false);
    renderMessages();
  }
}

async function refreshFiles() {
  if (!st.activeId) return;
  try {
    const resp = await fetch(`/session/${st.activeId}/files`);
    if (resp.ok) {
      const data = await resp.json();
      st.files = data.files || [];
      renderFiles();
    }
  } catch (e) {
    console.warn("Failed to load files:", e);
  }
}

/* ── Render Sessions ── */
function renderSessions() {
  const l = el.sList;
  if (!st.sessions.length) {
    l.innerHTML = '<div class="sidebar__empty">Upload a CSV to start</div>';
    return;
  }
  let h = "";
  for (const s of st.sessions) {
    const on = s.id === st.activeId;
    h += `<button class="session-item${on ? " active" : ""}" data-id="${s.id}">
      <span class="session-item__icon">📊</span>
      <span class="session-item__text">${esc(s.filename)}<br><small>${s.rows} rows · ${s.columns} cols</small></span>
    </button>`;
  }
  l.innerHTML = h;
  l.querySelectorAll(".session-item").forEach(el => {
    el.addEventListener("click", () => switchTo(parseInt(el.dataset.id)));
  });
}

function switchTo(id) {
  st.activeId = id;
  el.input.disabled = false;
  // We don't persist messages across sessions in-memory, so just clear for the switched session
  // In a more complete implementation, we'd cache messages per session
  renderSessions();
  refreshFiles();
  closeSidebar();
}

/* ── Render Messages ── */
function renderMessages() {
  if (!st.messages.length) {
    el.welcome.style.display = "";
    el.chatArea.style.display = "none";
    return;
  }
  el.welcome.style.display = "none";
  el.chatArea.style.display = "";
  let h = "";
  for (const m of st.messages) {
    if (m.role === "user") {
      h += `<div class="msg msg--user">
        <div class="msg__avatar-wrap msg__avatar-wrap--user"></div>
        <div class="msg__bubble nebula-card msg__bubble--user"><div class="msg__content">${fmt(m.content)}</div></div>
      </div>`;
    } else {
      const contentHtml = fmt(m.content);
      // Check for image references in the content and insert inline previews
      const enrichedHtml = enrichWithImages(contentHtml);
      
      h += `<div class="msg msg--ai">
        <div class="msg__avatar-wrap nebula-icon-bg"><div class="msg__avatar-emoji">🤖</div></div>
        <div class="msg__bubble nebula-card msg__bubble--ai">
          <div class="msg__content">${enrichedHtml}</div>
          <div class="msg__actions">
            <button class="msg__action-btn copy-btn" title="Copy" data-copy="${escA(m.content)}"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>
          </div>
        </div>
      </div>`;
    }
  }
  el.msgs.innerHTML = h;
  el.msgs.querySelectorAll(".copy-btn").forEach(b => {
    b.addEventListener("click", () => {
      navigator.clipboard.writeText(b.dataset.copy).then(() => {
        b.style.color = "#4ade80";
        setTimeout(() => b.style.color = "", 1500);
      });
    });
  });
  scrollDown();
}

/* Enrich response HTML with inline image previews for any mentioned plot files */
function enrichWithImages(html) {
  if (!st.activeId) return html;
  // Match common plot filenames the agent mentions
  const plotPatterns = /\b(plot_\w+\.png|predictions\.csv|report\.txt|model\.joblib|[\w]+\.png)\b/g;
  return html.replace(plotPatterns, (match) => {
    if (match.endsWith(".png")) {
      const url = `/session/${st.activeId}/files/${match}`;
      return `${match}<div class="inline-image"><img src="${url}" alt="${match}" loading="lazy" onclick="window.open('${url}','_blank')"/></div>`;
    }
    return match;
  });
}

/* ── Render Files Panel ── */
function renderFiles() {
  if (!st.files.length) {
    el.filesPanel.style.display = "none";
    return;
  }
  el.filesPanel.style.display = "";
  let h = "";
  for (const f of st.files) {
    const icon = getFileIcon(f.name);
    const sizeStr = formatSize(f.size);
    const url = `/session/${st.activeId}/files/${f.name}`;
    
    h += `<a href="${url}" target="_blank" download class="file-card">
      <span class="file-card__icon">${icon}</span>
      <span class="file-card__info">
        <span class="file-card__name">${esc(f.name)}</span>
        <span class="file-card__size">${sizeStr}</span>
      </span>
      <span class="file-card__dl">⬇</span>
    </a>`;
  }
  el.filesList.innerHTML = h;
}

function getFileIcon(name) {
  if (name.endsWith(".png") || name.endsWith(".jpg")) return "📊";
  if (name.endsWith(".csv")) return "📋";
  if (name.endsWith(".joblib")) return "🧠";
  if (name.endsWith(".txt")) return "📝";
  return "📄";
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

/* ── Sidebar ── */
function toggleSidebar() {
  if (window.innerWidth >= 1024) {
    el.sidebar.classList.toggle("collapsed");
  } else {
    if (el.sidebar.classList.contains("open")) closeSidebar();
    else openSidebar();
  }
}
function openSidebar() {
  if (window.innerWidth >= 1024) {
    el.sidebar.classList.remove("collapsed");
  } else {
    el.sidebar.classList.add("open");
    el.overlay.classList.add("open");
  }
}
function closeSidebar() {
  if (window.innerWidth >= 1024) {
    el.sidebar.classList.add("collapsed");
  } else {
    el.sidebar.classList.remove("open");
    el.overlay.classList.remove("open");
  }
}

/* ── Input ── */
function setupInput() {
  el.input.addEventListener("input", () => {
    el.input.style.height = "auto";
    el.input.style.height = Math.min(el.input.scrollHeight, window.innerHeight * 0.5) + "px";
    el.btnSend.disabled = !el.input.value.trim() || !st.activeId;
  });
  el.input.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  el.btnSend.addEventListener("click", send);
}

function showTyping(on) { el.typing.style.display = on ? "" : "none"; if (on) scrollDown(); }
function scrollDown() { requestAnimationFrame(() => { el.chatArea.scrollTop = el.chatArea.scrollHeight; }); }

/* ── Suggestion cards ── */
function setupCards() {
  document.querySelectorAll(".suggestion-card").forEach(c => {
    c.addEventListener("click", () => {
      if (!st.activeId) {
        // Trigger upload if no session
        el.fileInput.click();
        return;
      }
      el.input.value = c.dataset.q;
      el.input.dispatchEvent(new Event("input"));
      el.input.focus();
    });
  });
}

/* ── Files panel toggle ── */
function setupFilesPanel() {
  el.btnToggleFiles.addEventListener("click", () => {
    el.filesList.classList.toggle("collapsed");
    el.btnToggleFiles.textContent = el.filesList.classList.contains("collapsed") ? "▶" : "▼";
  });
}

/* ── Helpers ── */
function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
function escA(s) { return s.replace(/"/g, "&quot;").replace(/</g, "&lt;"); }
function fmt(t) {
  if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true });
    return marked.parse(t);
  }
  let h = esc(t);
  h = h.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  h = h.replace(/\n/g, "<br>");
  return h;
}

/* ── Boot ── */
function boot() {
  setupUpload();
  setupInput();
  setupCards();
  setupFilesPanel();
  el.btnMenu.addEventListener("click", toggleSidebar);
  el.overlay.addEventListener("click", closeSidebar);
  
  el.chatArea.addEventListener("click", () => {
    if (window.innerWidth < 1024 && el.sidebar.classList.contains("open")) closeSidebar();
  });
  el.input.addEventListener("focus", () => {
    if (window.innerWidth < 1024 && el.sidebar.classList.contains("open")) closeSidebar();
  });
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
