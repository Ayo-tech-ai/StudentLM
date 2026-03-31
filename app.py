import os
import streamlit as st
from fpdf import FPDF

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Study Assistant", page_icon="📘")

st.title("📘 AI Study Assistant")
st.caption("Understand your course materials step-by-step. No shortcuts, just learning.")

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

if "quiz_question" not in st.session_state:
    st.session_state.quiz_question = None

if "mode" not in st.session_state:
    st.session_state.mode = None

# NEW STATES
if "mcqs" not in st.session_state:
    st.session_state.mcqs = None

if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}

if "show_results" not in st.session_state:
    st.session_state.show_results = False

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader(
    "Upload your course material (PDF, DOCX, TXT):",
    type=["pdf", "docx", "txt"]
)

if uploaded_file is not None:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    # --- LOAD DOCUMENT ---
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

    # --- CREATE RETRIEVER ---
    st.session_state.retriever = FAISS.from_documents(docs, embeddings).as_retriever()

    st.success(f"✅ Document '{uploaded_file.name}' loaded!")

    # --- LIMIT TEXT FOR PROCESSING ---
    limited_docs = docs[:5]
    full_text = " ".join([doc.page_content for doc in limited_docs])

    # --- ACADEMIC SUMMARY ---
    if st.session_state.doc_summary is None:
        with st.spinner("Generating academic summary..."):
            summary_prompt = f"""
            Provide an academic summary of this document.

            Include:
            - Main topic
            - Key arguments
            - Important concepts
            - Conclusions

            Document:
            {full_text}
            """
            st.session_state.doc_summary = llm.invoke(summary_prompt).content

    # --- SECTION EXTRACTION ---
    if st.session_state.sections is None:
        with st.spinner("Structuring document into sections..."):
            section_prompt = f"""
            Divide this document into clear learning sections.

            For each section:
            - Give a short title
            - Provide the content

            Format strictly as:
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
                    title = title_split[0].strip()
                    content = title_split[1].strip()
                    sections.append({"title": title, "content": content})

            st.session_state.sections = sections

# --- DISPLAY SUMMARY ---
if st.session_state.doc_summary:
    st.markdown("### 📄 Academic Summary")
    st.info(st.session_state.doc_summary)

    col1, col2, col3, col4 = st.columns(4)

    if col1.button("📘 Learn"):
        st.session_state.mode = "learn"

    if col2.button("🧠 Key Ideas"):
        st.session_state.mode = "key"

    if col3.button("🎯 Practice"):
        st.session_state.mode = "practice"

    if col4.button("⚡ Exam Cram"):
        st.session_state.mode = "exam"

# --- SECTION NAVIGATION ---
if st.session_state.sections:
    st.markdown("### 📚 Study by Sections")

    section_titles = [sec["title"] for sec in st.session_state.sections]
    selected = st.selectbox("Choose a section:", section_titles)

    for sec in st.session_state.sections:
        if sec["title"] == selected:
            st.session_state.selected_section = sec

# --- SECTION LEARNING MODES ---
if st.session_state.selected_section:
    sec = st.session_state.selected_section

    st.markdown(f"## 📖 {sec['title']}")

    # --- LEARN ---
    if st.session_state.mode == "learn":
        with st.spinner("Explaining..."):
            explain_prompt = f"""
            Explain this section like a lecturer teaching a beginner.

            Use simple language and examples.

            Section:
            {sec['content']}
            """
            explanation = llm.invoke(explain_prompt).content
            st.write(explanation)

    # --- KEY IDEAS ---
    elif st.session_state.mode == "key":
        with st.spinner("Extracting key ideas..."):
            key_prompt = f"""
            Extract 5 key points from this section.

            Keep them short and clear.

            Section:
            {sec['content']}
            """
            points = llm.invoke(key_prompt).content
            st.write(points)

    # --- PRACTICE (INTERACTIVE MCQ) ---
    elif st.session_state.mode == "practice":

        if st.session_state.mcqs is None:
            with st.spinner("Generating questions..."):
                mcq_prompt = f"""
                Generate exactly 5 multiple choice questions from the section below.

                Format strictly as:

                Question: ...
                A) ...
                B) ...
                C) ...
                D) ...
                Answer: ...
                Explanation: ...

                Section:
                {sec['content']}
                """
                raw_mcqs = llm.invoke(mcq_prompt).content

                questions = raw_mcqs.split("Question:")
                mcqs = []

                for q in questions[1:]:
                    lines = q.strip().split("\n")
                    question_text = lines[0]

                    options = {}
                    answer = ""
                    explanation = ""

                    for line in lines:
                        if line.startswith("A)"):
                            options["A"] = line[2:].strip()
                        elif line.startswith("B)"):
                            options["B"] = line[2:].strip()
                        elif line.startswith("C)"):
                            options["C"] = line[2:].strip()
                        elif line.startswith("D)"):
                            options["D"] = line[2:].strip()
                        elif line.startswith("Answer:"):
                            answer = line.replace("Answer:", "").strip()
                        elif line.startswith("Explanation:"):
                            explanation = line.replace("Explanation:", "").strip()

                    mcqs.append({
                        "question": question_text,
                        "options": options,
                        "answer": answer,
                        "explanation": explanation
                    })

                st.session_state.mcqs = mcqs

        st.markdown("### 🎯 Practice Questions")

        for i, q in enumerate(st.session_state.mcqs):
            st.markdown(f"**Q{i+1}. {q['question']}**")

            choice = st.radio(
                f"Select answer for Q{i+1}",
                options=["A", "B", "C", "D"],
                key=f"q_{i}"
            )

            st.session_state.user_answers[i] = choice

        if st.button("Submit Answers"):
            st.session_state.show_results = True

        if st.session_state.show_results:
            st.markdown("## 📊 Results")

            for i, q in enumerate(st.session_state.mcqs):
                user_ans = st.session_state.user_answers.get(i)
                correct = q["answer"]

                if user_ans == correct:
                    st.success(f"Q{i+1}: Correct ✅")
                else:
                    st.error(f"Q{i+1}: Incorrect ❌")

                st.write(f"**Correct Answer:** {correct}")
                st.write(f"**Explanation:** {q['explanation']}")
                st.markdown("---")

            if st.button("⚡ Revise with Exam Cram"):
                st.session_state.mode = "exam"

    # --- EXAM CRAM ---
    elif st.session_state.mode == "exam":
        with st.spinner("Generating revision notes..."):
            cram_prompt = f"""
            Create quick revision notes for exam preparation.

            Include:
            - Key concepts
            - Definitions
            - Important facts
            - Likely exam questions

            Keep it concise and skimmable.

            Section:
            {sec['content']}
            """
            cram = llm.invoke(cram_prompt).content

        st.write(cram)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for line in cram.split("\n"):
            pdf.multi_cell(0, 8, line)

        pdf_file = "exam_cram.pdf"
        pdf.output(pdf_file)

        with open(pdf_file, "rb") as f:
            st.download_button(
                "📥 Download Exam Cram PDF",
                f,
                file_name="exam_cram.pdf",
                mime="application/pdf"
            )
