import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="StudyLM", page_icon="📘", layout="wide")

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Nunito:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

.stApp {
    background: #f0f7ff;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0369a1 0%, #0ea5e9 100%);
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label { color: white !important; }

/* ── Main heading ── */
.main-title {
    font-family: 'Sora', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #0369a1;
    margin-bottom: 0;
}
.main-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 0.1rem;
    margin-bottom: 1.5rem;
}

/* ── Chat bubbles ── */
.bubble-wrapper { display: flex; margin-bottom: 14px; align-items: flex-start; }
.bubble-wrapper.user  { flex-direction: row-reverse; }
.bubble-wrapper.tutor { flex-direction: row; }

.avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
    margin: 0 8px;
}
.avatar.tutor { background: #0ea5e9; }
.avatar.user  { background: #f59e0b; }

.bubble {
    max-width: 75%;
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 0.93rem;
    line-height: 1.55;
}
.bubble.tutor {
    background: white;
    border: 1px solid #e2e8f0;
    border-top-left-radius: 4px;
    color: #1e293b;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.bubble.user {
    background: #0369a1;
    color: white;
    border-top-right-radius: 4px;
}

/* ── Hint card ── */
.hint-card {
    background: #fefce8;
    border-left: 4px solid #eab308;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0 16px 52px;
    font-size: 0.88rem;
    color: #713f12;
}

/* ── Status badge ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 12px;
}
.badge.loaded { background: #dcfce7; color: #166534; }
.badge.empty  { background: #fee2e2; color: #991b1b; }

/* ── Buttons ── */
.stButton > button {
    background: #0369a1 !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    background: #0284c7 !important;
}

/* ── Chat input ── */
.stChatInput textarea {
    border-radius: 12px !important;
    border: 1.5px solid #bae6fd !important;
    font-family: 'Nunito', sans-serif !important;
}

/* ── Divider ── */
hr { border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ── LLM SETUP ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
embeddings = HuggingFaceEmbeddings()

# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are StudyLM, a warm and encouraging Socratic tutor for Nigerian Junior Secondary School (JSS1–JSS3) students studying {subject}.

Your core philosophy:
- NEVER give direct answers to assignment questions or homework problems.
- ALWAYS guide the student to discover the answer themselves through questions, hints, and encouragement.
- Speak simply and clearly, as if talking to a 12–15 year old Nigerian student.
- Be warm, patient, and celebratory of small wins ("Great thinking!", "You're on the right track!").

When a student uploads a document or pastes assignment questions and asks you to answer them directly:
1. Kindly acknowledge what they shared.
2. Explain warmly that you won't do the assignment for them.
3. Ask them: "Before we dive in, what do you already know about this topic?"
4. Then guide them step by step using questions and hints.

When a student genuinely wants to understand a concept:
1. Ask what they already know first.
2. Use real-life Nigerian examples where possible (e.g., Lagos traffic for Geography, local plants for Biology).
3. Break concepts into small, digestible questions.
4. Only give a hint when the student is stuck — never the full answer.
5. Celebrate progress at every step.

Detecting assignment dumping:
- If the message looks like a list of questions, an exam paper, or a structured assignment, assume the student is trying to get answers. Redirect kindly.
- If the student says "answer this", "solve this", "do this for me", "what is the answer to", redirect warmly.

Document context will be provided when available. Use it to base your guidance on the actual material the student is studying.

Remember: Your job is to make them THINK, not to think FOR them.
"""


# ── SESSION STATE ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],          # {"role": "tutor"/"user", "content": "..."}
        "retriever": None,
        "doc_loaded": False,
        "doc_name": None,
        "subject": "Biology",
        "hint_pending": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── HELPERS ────────────────────────────────────────────────────────────────────
def get_context(query: str) -> str:
    """Retrieve relevant chunks from the uploaded document."""
    if st.session_state.retriever is None:
        return ""
    try:
        docs = st.session_state.retriever.get_relevant_documents(query)
        return "\n\n".join([d.page_content for d in docs[:3]])
    except Exception:
        return ""


def build_messages(user_query: str) -> list:
    """Build the full message list for the LLM call."""
    context = get_context(user_query)
    system = SYSTEM_PROMPT.format(subject=st.session_state.subject)
    if context:
        system += f"\n\n--- Relevant content from the student's document ---\n{context}\n---"

    messages = [{"role": "system", "content": system}]

    # Add conversation history (last 10 turns to stay within context limits)
    for msg in st.session_state.messages[-20:]:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})

    messages.append({"role": "user", "content": user_query})
    return messages


def call_llm(messages: list) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    lc_messages = []
    for m in messages:
        if m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))
    return llm.invoke(lc_messages).content


def render_chat():
    for msg in st.session_state.messages:
        role = msg["role"]
        icon = "📘" if role == "tutor" else "🧑‍🎓"
        st.markdown(f"""
        <div class="bubble-wrapper {role}">
            <div class="avatar {role}">{icon}</div>
            <div class="bubble {role}">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📘 StudyLM")
    st.markdown("*Your Socratic study companion*")
    st.markdown("---")

    # Subject picker
    subject = st.selectbox(
        "📚 Subject",
        ["Biology", "English", "Geography"],
        index=["Biology", "English", "Geography"].index(st.session_state.subject)
    )
    if subject != st.session_state.subject:
        st.session_state.subject = subject

    st.markdown("---")

    # File upload
    st.markdown("**📂 Upload Your Material**")
    uploaded_file = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if st.session_state.doc_name != uploaded_file.name:
            with st.spinner("Reading document..."):
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                if uploaded_file.name.endswith(".pdf"):
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(temp_path)
                elif uploaded_file.name.endswith(".docx"):
                    from langchain_community.document_loaders import Docx2txtLoader
                    loader = Docx2txtLoader(temp_path)
                else:
                    from langchain_community.document_loaders import TextLoader
                    loader = TextLoader(temp_path)

                docs = loader.load()
                st.session_state.retriever = FAISS.from_documents(docs, embeddings).as_retriever()
                st.session_state.doc_loaded = True
                st.session_state.doc_name = uploaded_file.name

                # Inject tutor greeting about the doc
                welcome = call_llm([
                    {"role": "system", "content": SYSTEM_PROMPT.format(subject=subject)},
                    {"role": "user", "content": f"I just uploaded a document called '{uploaded_file.name}'. Please greet me and ask me what I want to learn or understand from it — but do NOT summarise or answer anything yet."}
                ])
                st.session_state.messages.append({"role": "tutor", "content": welcome})

    # Doc status
    if st.session_state.doc_loaded:
        st.markdown(f'<span class="badge loaded">✅ {st.session_state.doc_name}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge empty">No document uploaded</span>', unsafe_allow_html=True)

    st.markdown("---")

    # How it works
    st.markdown("**💡 How StudyLM works**")
    st.markdown("""
- 📤 Upload your notes or assignment
- 💬 Tell me what you want to understand
- 🧠 I'll guide you with questions & hints
- ✅ You learn by *thinking*, not copying
    """)

    st.markdown("---")
    if st.button("🔄 Start Fresh"):
        for k in ["messages", "retriever", "doc_loaded", "doc_name"]:
            if k == "messages":
                st.session_state.messages = []
            else:
                st.session_state[k] = None if k != "doc_loaded" else False
        st.rerun()


# ── MAIN AREA ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📘 StudyLM</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Your Socratic tutor — I help you <em>understand</em>, not just get answers.</div>', unsafe_allow_html=True)

# First-time greeting (no doc uploaded)
if not st.session_state.messages and not st.session_state.doc_loaded:
    intro = f"""👋 Hello! I'm StudyLM, your personal study tutor for **{st.session_state.subject}**.

I'm a little different from other AI tools — I won't just give you answers. Instead, I'll ask you questions and drop hints so *you* can figure things out yourself. That's how real learning happens! 😊

Here's how to get started:
- 📤 **Upload a document** (your notes, textbook pages, or assignment) using the sidebar
- 💬 Or just **type a topic** you want to understand

What would you like to learn today?"""
    st.session_state.messages.append({"role": "tutor", "content": intro})

# Render chat history
render_chat()

# Chat input
user_input = st.chat_input("Ask me something, or tell me what you want to understand...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get tutor response
    with st.spinner("Thinking..."):
        messages = build_messages(user_input)
        response = call_llm(messages)

    st.session_state.messages.append({"role": "tutor", "content": response})
    st.rerun()
