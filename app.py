import os
import streamlit as st
from fpdf import FPDF

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- PAGE CONFIG ---
st.set_page_config(page_title="StudyLM", page_icon="📘", layout="wide")

# --- CUSTOM CSS (UI MAGIC) ---
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.card-title {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 10px;
}
button[kind="primary"] {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("## 📘 StudyLM")
st.caption("Your AI-powered study companion. Learn smarter, not harder.")

# --- API ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

embeddings = HuggingFaceEmbeddings()

# --- SESSION STATE ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = None
if "sections" not in st.session_state:
    st.session_state.sections = None
if "selected_section" not in st.session_state:
    st.session_state.selected_section = None
if "mode" not in st.session_state:
    st.session_state.mode = None
if "mcqs" not in st.session_state:
    st.session_state.mcqs = None
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# --- FILE UPLOAD ---
st.markdown('<div class="card"><div class="card-title">📂 Upload Document</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload your course material (PDF, DOCX, TXT):",
    type=["pdf", "docx", "txt"]
)

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    if uploaded_file.name.endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
    elif uploaded_file.name.endswith(".docx"):
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(temp_file_path)
    else:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(temp_file_path)

    docs = loader.load()

    st.session_state.retriever = FAISS.from_documents(docs, embeddings).as_retriever()
    st.success(f"✅ {uploaded_file.name} loaded successfully!")

    limited_docs = docs[:5]
    full_text = " ".join([doc.page_content for doc in limited_docs])

    # --- SUMMARY ---
    if st.session_state.doc_summary is None:
        with st.spinner("Generating summary..."):
            summary_prompt = f"""
            Provide an academic summary of this document.
            Include main topic, key arguments, concepts, and conclusions.
            Document:
            {full_text}
            """
            st.session_state.doc_summary = llm.invoke(summary_prompt).content

    # --- SECTIONS ---
    if st.session_state.sections is None:
        with st.spinner("Structuring document..."):
            section_prompt = f"""
            Divide this document into sections.
            Format:
            Title: ...
            Content: ...
            Document:
            {full_text}
            """
            raw_sections = llm.invoke(section_prompt).content

            sections = []
            parts = raw_sections.split("Title:")
            for part in parts[1:]:
                title_split = part.split("Content:")
                if len(title_split) == 2:
                    sections.append({
                        "title": title_split[0].strip(),
                        "content": title_split[1].strip()
                    })
            st.session_state.sections = sections

# --- SUMMARY DISPLAY ---
if st.session_state.doc_summary:
    st.markdown('<div class="card"><div class="card-title">📄 Academic Summary</div>', unsafe_allow_html=True)
    st.write(st.session_state.doc_summary)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- MODES ---
    st.markdown("### 🎯 Choose Study Mode")
    col1, col2, col3, col4 = st.columns(4)

    if col1.button("📘 Learn"):
        st.session_state.mode = "learn"
    if col2.button("🧠 Key Ideas"):
        st.session_state.mode = "key"
    if col3.button("🎯 Practice"):
        st.session_state.mode = "practice"
    if col4.button("⚡ Exam Cram"):
        st.session_state.mode = "exam"

# --- SECTION SELECT ---
if st.session_state.sections:
    st.markdown('<div class="card"><div class="card-title">📚 Study Sections</div>', unsafe_allow_html=True)

    section_titles = [sec["title"] for sec in st.session_state.sections]
    selected = st.selectbox("Select a section:", section_titles)

    for sec in st.session_state.sections:
        if sec["title"] == selected:
            st.session_state.selected_section = sec

    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN CONTENT ---
if st.session_state.selected_section:
    sec = st.session_state.selected_section

    st.markdown(f"## 📖 {sec['title']}")

    # --- LEARN ---
    if st.session_state.mode == "learn":
        with st.spinner("Explaining..."):
            explanation = llm.invoke(f"Explain simply:\n{sec['content']}").content
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(explanation)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- KEY IDEAS ---
    elif st.session_state.mode == "key":
        with st.spinner("Extracting..."):
            points = llm.invoke(f"Give 5 key points:\n{sec['content']}").content
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(points)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- PRACTICE ---
    elif st.session_state.mode == "practice":

        if st.session_state.mcqs is None:
            with st.spinner("Generating questions..."):
                mcqs = llm.invoke(f"Generate 5 MCQs:\n{sec['content']}").content
                st.session_state.mcqs = mcqs

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(st.session_state.mcqs)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("⚡ Revise with Exam Cram"):
            st.session_state.mode = "exam"

    # --- EXAM CRAM ---
    elif st.session_state.mode == "exam":
        with st.spinner("Generating notes..."):
            cram = llm.invoke(f"Create revision notes:\n{sec['content']}").content

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(cram)
        st.markdown('</div>', unsafe_allow_html=True)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for line in cram.split("\n"):
            pdf.multi_cell(0, 8, line)

        pdf.output("exam_cram.pdf")

        with open("exam_cram.pdf", "rb") as f:
            st.download_button("📥 Download PDF", f)
