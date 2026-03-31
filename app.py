import os
import streamlit as st
from fpdf import FPDF

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Study Assistant", page_icon="📘")

st.title("📘 AI Study Assistant")
st.caption("Study smarter. Understand, practice, and revise effectively.")

# --- API ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

embeddings = OpenAIEmbeddings()

# --- SESSION STATE ---
for key in ["retriever", "doc_summary", "sections", "selected_section", "mode", "exam_cram"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your course material:", type=["pdf", "docx", "txt"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load document
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

    full_text = " ".join([d.page_content for d in docs[:5]])

    # --- SUMMARY ---
    if not st.session_state.doc_summary:
        with st.spinner("Generating summary..."):
            st.session_state.doc_summary = llm.invoke(f"""
            Provide a concise academic summary.

            {full_text}
            """).content

    # --- SECTIONS ---
    if not st.session_state.sections:
        with st.spinner("Structuring sections..."):
            raw = llm.invoke(f"""
            Divide into sections.

            Format:
            Title: ...
            Content: ...

            {full_text}
            """).content

            sections = []
            parts = raw.split("Title:")
            for p in parts[1:]:
                t, c = p.split("Content:")
                sections.append({"title": t.strip(), "content": c.strip()})

            st.session_state.sections = sections

# --- SUMMARY DISPLAY ---
if st.session_state.doc_summary:
    st.markdown("### 📄 Academic Summary")
    st.info(st.session_state.doc_summary)

    # --- MODE BUTTONS ---
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
    titles = [s["title"] for s in st.session_state.sections]
    selected = st.selectbox("Choose a section:", titles)

    for s in st.session_state.sections:
        if s["title"] == selected:
            st.session_state.selected_section = s

# --- MODES ---
if st.session_state.selected_section:
    sec = st.session_state.selected_section

    st.markdown(f"## 📖 {sec['title']}")

    # --- LEARN ---
    if st.session_state.mode == "learn":
        with st.spinner("Teaching..."):
            res = llm.invoke(f"Explain simply:\n{sec['content']}")
            st.write(res.content)

    # --- KEY IDEAS ---
    elif st.session_state.mode == "key":
        with st.spinner("Extracting..."):
            res = llm.invoke(f"Give 5 key bullet points:\n{sec['content']}")
            st.write(res.content)

    # --- PRACTICE ---
    elif st.session_state.mode == "practice":
        with st.spinner("Generating MCQs..."):
            mcq = llm.invoke(f"""
            Generate 3 MCQs with options A-D and correct answers.

            {sec['content']}
            """).content

        st.write(mcq)

        # Learning Loop
        st.markdown("---")
        if st.button("⚡ Revise with Exam Cram"):
            st.session_state.mode = "exam"

    # --- EXAM CRAM ---
    elif st.session_state.mode == "exam":
        if not st.session_state.exam_cram:
            with st.spinner("Creating revision notes..."):
                cram = llm.invoke(f"""
                Create exam revision notes:

                - Key concepts
                - Definitions
                - Likely questions

                {sec['content']}
                """).content

                st.session_state.exam_cram = cram

        st.write(st.session_state.exam_cram)

        # --- PDF DOWNLOAD ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for line in st.session_state.exam_cram.split("\n"):
            pdf.multi_cell(0, 8, line)

        pdf_path = "exam_cram.pdf"
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download PDF", f, file_name="exam_cram.pdf")
