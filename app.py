import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="StudyLM", page_icon="📘", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Nunito:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
.stApp { background: #f0f7ff; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0369a1 0%, #0ea5e9 100%);
}
[data-testid="stSidebar"] * { color: white !important; }

.main-title {
    font-family: 'Sora', sans-serif;
    font-size: 2rem; font-weight: 700;
    color: #0369a1; margin-bottom: 0;
}
.main-subtitle {
    color: #64748b; font-size: 0.95rem;
    margin-top: 0.1rem; margin-bottom: 1rem;
}

/* ── Progress bar ── */
.progress-bar-wrap {
    background: #e0f2fe; border-radius: 20px;
    height: 12px; margin: 6px 0 2px 0; overflow: hidden;
}
.progress-bar-fill {
    background: linear-gradient(90deg, #0ea5e9, #0369a1);
    height: 100%; border-radius: 20px;
    transition: width 0.5s ease;
}
.progress-label {
    font-size: 0.78rem; color: #0369a1;
    font-weight: 600; margin-bottom: 12px;
}

/* ── Concept chips ── */
.concept-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 16px 0; }
.chip {
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
}
.chip.done  { background: #dcfce7; color: #166534; }
.chip.active { background: #fef9c3; color: #854d0e; }
.chip.todo  { background: #f1f5f9; color: #94a3b8; }

/* ── Chat bubbles ── */
.bubble-wrapper { display: flex; margin-bottom: 14px; align-items: flex-start; }
.bubble-wrapper.user  { flex-direction: row-reverse; }
.bubble-wrapper.tutor { flex-direction: row; }
.avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0; margin: 0 8px;
}
.avatar.tutor { background: #0ea5e9; }
.avatar.user  { background: #f59e0b; }
.bubble {
    max-width: 75%; padding: 12px 16px;
    border-radius: 16px; font-size: 0.93rem; line-height: 1.6;
}
.bubble.tutor {
    background: white; border: 1px solid #e2e8f0;
    border-top-left-radius: 4px; color: #1e293b;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.bubble.user {
    background: #0369a1; color: white;
    border-top-right-radius: 4px;
}

/* ── Action buttons row ── */
.action-row { display: flex; gap: 8px; margin: 0 0 16px 52px; flex-wrap: wrap; }

/* ── Session summary card ── */
.summary-card {
    background: white; border: 2px solid #0ea5e9;
    border-radius: 14px; padding: 18px 20px; margin: 16px 0;
}
.summary-card h4 { color: #0369a1; margin: 0 0 10px 0; font-family: 'Sora', sans-serif; }

/* ── Badge ── */
.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; margin-bottom: 12px; }
.badge.loaded { background: #dcfce7; color: #166534; }
.badge.empty  { background: #fee2e2; color: #991b1b; }

.stButton > button {
    background: #0369a1 !important; color: white !important;
    border-radius: 8px !important; border: none !important;
    font-family: 'Nunito', sans-serif !important; font-weight: 600 !important;
}
.stButton > button:hover { background: #0284c7 !important; }

.stChatInput textarea {
    border-radius: 12px !important;
    border: 1.5px solid #bae6fd !important;
    font-family: 'Nunito', sans-serif !important;
}
hr { border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ── LLM SETUP ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
embeddings = HuggingFaceEmbeddings()


# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are StudyLM, a friendly Socratic tutor for Nigerian JSS1–JSS3 students studying {subject}.

LANGUAGE RULES — very important:
- Use simple, short sentences. Max 2–3 sentences per paragraph.
- Speak like a friendly Nigerian teacher. Use phrases like "Oya, let's think about this", "Well done!", "You're getting there!", "No wahala, let's try again" — naturally, not forced.
- NEVER use big grammar words like "etymological", "stimuli", "stationary". Use everyday words a 13-year-old knows.
- Ask only ONE question at a time. Never two questions in one message.

STUCK DETECTION — critical:
- Track how many times the student has tried to answer the current concept without getting it right.
- Stuck level is tracked in the conversation. After each wrong/incomplete answer, the hint gets bigger:
  * Attempt 1 wrong: Ask a simpler leading question.
  * Attempt 2 wrong: Give a small direct hint (one sentence clue).
  * Attempt 3 wrong: Give a bigger hint — almost tell them, but leave one word for them to fill in.
  * Attempt 4+ wrong: Gently explain that part simply, then move on. Say "No wahala — this one is tricky. Here's the simple truth: [explain simply]. Now let's keep going!"

ANSWER CONFIRMATION — critical:
- When a student says something like "I think the answer is...", "My answer is...", "Is it...?", "So the answer is..." — this means they want to confirm their answer.
- Check if their answer is correct based on the document context and your knowledge.
- If CORRECT: Celebrate warmly! ("Yes! That's exactly it! Well done! 🎉") Then move to the next concept.
- If PARTIALLY CORRECT: Tell them what part is right, and ask one question to help them fix the missing part.
- If WRONG: Don't say "wrong." Say "Hmm, not quite — but you're thinking! Let me give you a small clue..." then give a hint.

STRUCTURE VISIBILITY — always do this:
- When starting a new topic, tell the student how many concepts/points you'll cover. E.g. "This topic has 5 key ideas. Let's tackle them one by one. We're starting with number 1!"
- After each concept is understood, say: "✅ That's concept 1 done! Moving to concept 2 now..."
- This makes the student feel progress.

SIMPLIFY MODE:
- If the student says "I don't understand", "simplify", "explain again", or "what do you mean" — rephrase your last explanation using even simpler words and a real-life Nigerian example (food, market, school, nature in Nigeria).

ASSIGNMENT DUMPING:
- If the student pastes a list of questions or says "answer this / solve this / do this for me" — kindly say: "Ah ah! I can't do the assignment for you o 😄 But I'll help you understand it so well that you'll answer it yourself. Oya — what do you already know about this topic?"

CORE RULE: NEVER give a direct answer to an assignment question. Guide always. The student must do the thinking.

Document context will be provided below when available.
"""


# ── SESSION STATE ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "retriever": None,
        "doc_loaded": False,
        "doc_name": None,
        "subject": "Biology",
        "concepts_total": 0,
        "concepts_done": [],      # list of concept names completed
        "concepts_active": "",    # current concept being studied
        "concepts_todo": [],      # remaining concept names
        "stuck_count": 0,         # how many wrong attempts on current concept
        "show_summary": False,
        "simplify_requested": False,
        "confirm_requested": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── HELPERS ────────────────────────────────────────────────────────────────────
def get_context(query: str) -> str:
    if st.session_state.retriever is None:
        return ""
    try:
        docs = st.session_state.retriever.get_relevant_documents(query)
        return "\n\n".join([d.page_content for d in docs[:3]])
    except Exception:
        return ""


def build_messages(user_query: str, extra_instruction: str = "") -> list:
    context = get_context(user_query)
    system = SYSTEM_PROMPT.format(subject=st.session_state.subject)
    if context:
        system += f"\n\n--- Student's document content ---\n{context}\n---"
    if extra_instruction:
        system += f"\n\nSPECIAL INSTRUCTION FOR THIS TURN: {extra_instruction}"

    messages = [{"role": "system", "content": system}]
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


def generate_session_summary() -> str:
    concepts_done = st.session_state.concepts_done
    total = len(concepts_done)
    if total == 0:
        return "You're just getting started — no concepts completed yet. Keep going! 💪"
    prompt = f"""
    The student just finished a study session on {st.session_state.subject}.
    They successfully understood these concepts: {', '.join(concepts_done)}.
    Write a short, warm end-of-session summary (3–4 sentences max).
    Celebrate what they learned. Use simple, encouraging Nigerian-friendly language.
    End with: "Come back tomorrow to keep growing! 🌟"
    """
    return call_llm([{"role": "system", "content": "You are StudyLM."}, {"role": "user", "content": prompt}])


def render_progress():
    done = len(st.session_state.concepts_done)
    total = st.session_state.concepts_total
    if total == 0:
        return
    pct = int((done / total) * 100)
    st.markdown(f'<div class="progress-label">📊 Progress: {done} of {total} concepts understood ({pct}%)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:{pct}%"></div></div>', unsafe_allow_html=True)

    # Concept chips
    chips_html = '<div class="concept-row">'
    for c in st.session_state.concepts_done:
        chips_html += f'<span class="chip done">✅ {c}</span>'
    if st.session_state.concepts_active:
        chips_html += f'<span class="chip active">📖 {st.session_state.concepts_active}</span>'
    for c in st.session_state.concepts_todo:
        chips_html += f'<span class="chip todo">○ {c}</span>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)


def render_chat():
    for msg in st.session_state.messages:
        role = msg["role"]
        icon = "📘" if role == "tutor" else "🧑‍🎓"
        content = msg["content"].replace("\n", "<br>")
        st.markdown(f"""
        <div class="bubble-wrapper {role}">
            <div class="avatar {role}">{icon}</div>
            <div class="bubble {role}">{content}</div>
        </div>
        """, unsafe_allow_html=True)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📘 StudyLM")
    st.markdown("*Your Socratic study companion*")
    st.markdown("---")

    subject = st.selectbox(
        "📚 Subject",
        ["Biology", "English", "Geography"],
        index=["Biology", "English", "Geography"].index(st.session_state.subject)
    )
    if subject != st.session_state.subject:
        st.session_state.subject = subject

    st.markdown("---")
    st.markdown("**📂 Upload Your Material**")
    uploaded_file = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], label_visibility="collapsed")

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
                st.session_state.stuck_count = 0

                # Ask LLM to identify concepts in the doc for structure visibility
                preview_text = " ".join([d.page_content for d in docs[:3]])
                concept_prompt = f"""
                Read this document and list the main concepts or points it covers.
                Return ONLY a comma-separated list of concept names (max 8). No explanation.
                Document: {preview_text[:2000]}
                """
                raw_concepts = call_llm([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": concept_prompt}])
                concepts = [c.strip() for c in raw_concepts.split(",") if c.strip()]
                st.session_state.concepts_todo = concepts
                st.session_state.concepts_total = len(concepts)
                st.session_state.concepts_active = concepts[0] if concepts else ""

                welcome = call_llm([
                    {"role": "system", "content": SYSTEM_PROMPT.format(subject=subject)},
                    {"role": "user", "content": f"I uploaded '{uploaded_file.name}'. Greet me warmly, tell me this topic has {len(concepts)} key concepts we'll cover one by one, and ask what I already know about the first concept: '{concepts[0] if concepts else 'this topic'}'. Keep it short and friendly."}
                ])
                st.session_state.messages.append({"role": "tutor", "content": welcome})

    if st.session_state.doc_loaded:
        st.markdown(f'<span class="badge loaded">✅ {st.session_state.doc_name}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge empty">No document uploaded</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**💡 How it works**")
    st.markdown("""
- 📤 Upload your notes or assignment
- 💬 Tell me what you want to understand
- 🧠 I guide you — you do the thinking
- ✅ Confirm your answer when ready
- 🎯 Track your progress concept by concept
    """)
    st.markdown("---")

    # End session / summary
    if st.button("📋 End Session & See Summary"):
        st.session_state.show_summary = True
        st.rerun()

    if st.button("🔄 Start Fresh"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ── MAIN AREA ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📘 StudyLM</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Your Socratic tutor — I help you <em>understand</em>, not just get answers.</div>', unsafe_allow_html=True)

# Progress bar
render_progress()
st.markdown("---")

# Session summary
if st.session_state.show_summary:
    summary = generate_session_summary()
    st.markdown(f"""
    <div class="summary-card">
        <h4>🎓 Session Summary</h4>
        {summary.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)
    st.session_state.show_summary = False

# First-time greeting
if not st.session_state.messages:
    intro = f"""👋 Hello! I'm StudyLM, your study tutor for **{st.session_state.subject}**.

I'm different from other AI — I won't give you answers. Instead I'll ask you questions so *you* figure things out. That's how you really learn! 😊

To start:
- 📤 **Upload a document** from the sidebar (your notes or assignment)
- 💬 Or just **type a topic** you want to understand

What would you like to learn today?"""
    st.session_state.messages.append({"role": "tutor", "content": intro})

# Render chat
render_chat()

# ── ACTION BUTTONS ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    if st.button("🔍 Simplify this for me"):
        st.session_state.simplify_requested = True

with col2:
    if st.button("✅ Check my answer"):
        st.session_state.confirm_requested = True

with col3:
    if st.button("💡 I need a hint"):
        hint_msg = build_messages(
            "I am stuck and need a hint please.",
            extra_instruction=f"The student is stuck. This is stuck attempt number {st.session_state.stuck_count + 1}. Give an appropriately sized hint based on stuck level."
        )
        with st.spinner("Getting hint..."):
            hint_response = call_llm(hint_msg)
        st.session_state.stuck_count += 1
        st.session_state.messages.append({"role": "tutor", "content": f"💡 **Hint:** {hint_response}"})
        st.rerun()

# Handle simplify button
if st.session_state.simplify_requested:
    st.session_state.simplify_requested = False
    msgs = build_messages(
        "I don't understand. Please simplify.",
        extra_instruction="The student didn't understand your last message. Re-explain using even simpler words and one everyday Nigerian example. Keep it very short."
    )
    with st.spinner("Simplifying..."):
        simple_response = call_llm(msgs)
    st.session_state.messages.append({"role": "user", "content": "🔍 Please simplify that for me."})
    st.session_state.messages.append({"role": "tutor", "content": simple_response})
    st.rerun()

# Handle confirm answer button
if st.session_state.confirm_requested:
    st.session_state.confirm_requested = False
    user_answer = st.text_input("Type your answer here and press Enter:", key="answer_input")
    if user_answer:
        msgs = build_messages(
            f"I think my answer is: {user_answer}",
            extra_instruction="The student is submitting their answer for confirmation. Check if it is correct based on the document and topic. If correct, celebrate and mark this concept as done. If partially correct, praise what's right and ask one question to fix the rest. If wrong, don't say 'wrong' — say 'Hmm, not quite!' and give a clue."
        )
        with st.spinner("Checking your answer..."):
            confirm_response = call_llm(msgs)

        st.session_state.messages.append({"role": "user", "content": f"✅ My answer: {user_answer}"})
        st.session_state.messages.append({"role": "tutor", "content": confirm_response})

        # If answer confirmed correct, advance concepts
        if any(word in confirm_response.lower() for word in ["correct", "exactly", "well done", "that's right", "yes!", "perfect"]):
            st.session_state.stuck_count = 0
            if st.session_state.concepts_active:
                st.session_state.concepts_done.append(st.session_state.concepts_active)
            if st.session_state.concepts_todo:
                st.session_state.concepts_todo.pop(0)
                st.session_state.concepts_active = st.session_state.concepts_todo[0] if st.session_state.concepts_todo else ""

        st.rerun()

# ── CHAT INPUT ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Type here — ask a question, share what you know, or try to answer...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    extra = ""
    if st.session_state.stuck_count >= 2:
        extra = f"The student has been stuck {st.session_state.stuck_count} times. Increase your hint size accordingly."

    with st.spinner("Thinking..."):
        msgs = build_messages(user_input, extra_instruction=extra)
        response = call_llm(msgs)

    st.session_state.messages.append({"role": "tutor", "content": response})

    # Check if tutor confirmed a correct answer in the response
    if any(word in response.lower() for word in ["correct", "exactly", "well done", "that's right", "yes!", "perfect", "you got it"]):
        st.session_state.stuck_count = 0
        if st.session_state.concepts_active and st.session_state.concepts_active not in st.session_state.concepts_done:
            st.session_state.concepts_done.append(st.session_state.concepts_active)
            if st.session_state.concepts_todo:
                st.session_state.concepts_todo.pop(0)
                st.session_state.concepts_active = st.session_state.concepts_todo[0] if st.session_state.concepts_todo else ""
    else:
        st.session_state.stuck_count += 1

    st.rerun()
