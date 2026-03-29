import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
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

            # Simple parsing
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

    mode = st.radio(
        "Choose how you want to study:",
        ["Understand", "Key Points", "Test Me"]
    )

    # --- UNDERSTAND MODE ---
    if mode == "Understand":
        with st.spinner("Explaining..."):
            explain_prompt = f"""
            Explain this section like a lecturer teaching a beginner.

            Use simple language and examples.

            Section:
            {sec['content']}
            """
            explanation = llm.invoke(explain_prompt).content
            st.write(explanation)

    # --- KEY POINTS ---
    elif mode == "Key Points":
        with st.spinner("Extracting key points..."):
            key_prompt = f"""
            Extract 5 key points from this section.

            Keep them short and clear.

            Section:
            {sec['content']}
            """
            points = llm.invoke(key_prompt).content
            st.write(points)

    # --- TEST MODE ---
    elif mode == "Test Me":

        if st.session_state.quiz_question is None:
            with st.spinner("Generating question..."):
                q_prompt = f"""
                Generate 1 challenging question from this section.
                """
                st.session_state.quiz_question = llm.invoke(q_prompt).content

        st.write("### ❓ Question")
        st.write(st.session_state.quiz_question)

        user_answer = st.text_area("Your Answer:")

        if st.button("Submit Answer"):
            with st.spinner("Evaluating..."):
                eval_prompt = f"""
                Question: {st.session_state.quiz_question}

                Student Answer: {user_answer}

                Evaluate the answer.
                Give feedback and correct explanation.
                """
                feedback = llm.invoke(eval_prompt).content
                st.write("### 📊 Feedback")
                st.write(feedback)

            st.session_state.quiz_question = None
