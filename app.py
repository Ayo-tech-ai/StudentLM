import os
import streamlit as st
from fpdf import FPDF

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="StudyLM - AI Study Companion", 
    page_icon="📚", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (PROFESSIONAL UI) ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main container */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Card Styles */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2d3748;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    
    /* Mode Buttons Container */
    .mode-buttons {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Content Styles */
    .content-box {
        background: #f7fafc;
        border-radius: 12px;
        padding: 1.5rem;
        line-height: 1.6;
        color: #2d3748;
        margin-top: 1rem;
    }
    
    /* Success Message */
    .success-message {
        background: #c6f6d5;
        color: #22543d;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #48bb78;
        margin: 1rem 0;
    }
    
    /* Active Mode Indicator */
    .active-mode {
        background: #e6f7ff;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    /* Custom Streamlit Overrides */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Center content */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("""
<div class="main-header">
    <h1>📚 StudyLM</h1>
    <p>Your AI-powered study companion for smarter, faster learning</p>
</div>
""", unsafe_allow_html=True)

# --- API SETUP ---
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
if "upload_complete" not in st.session_state:
    st.session_state.upload_complete = False
if "explanation_content" not in st.session_state:
    st.session_state.explanation_content = None
if "key_points_content" not in st.session_state:
    st.session_state.key_points_content = None
if "exam_cram_content" not in st.session_state:
    st.session_state.exam_cram_content = None

# --- MAIN CONTENT (Centered) ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # --- FILE UPLOAD SECTION ---
    with st.container():
        st.markdown("""
        <div class="card">
            <div class="card-title">📂 Upload Your Study Material</div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Supporting PDF, DOCX, and TXT formats",
            type=["pdf", "docx", "txt"],
            help="Upload your course material to generate summaries and study materials"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing your document..."):
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
                st.session_state.upload_complete = True
                
                st.markdown(f"""
                <div class="success-message">
                    ✅ Successfully loaded <strong>{uploaded_file.name}</strong>! Ready to study.
                </div>
                """, unsafe_allow_html=True)
                
                limited_docs = docs[:5]
                full_text = " ".join([doc.page_content for doc in limited_docs])
                
                # --- GENERATE SUMMARY ---
                if st.session_state.doc_summary is None:
                    with st.spinner("Creating smart summary..."):
                        summary_prompt = f"""
                        Provide an academic summary of this document.
                        Include main topic, key arguments, concepts, and conclusions.
                        Document:
                        {full_text}
                        """
                        st.session_state.doc_summary = llm.invoke(summary_prompt).content
                
                # --- GENERATE SECTIONS ---
                if st.session_state.sections is None:
                    with st.spinner("Structuring content..."):
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

# --- STUDY CONTENT SECTION (Centered) ---
if st.session_state.upload_complete:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # --- SUMMARY DISPLAY ---
        if st.session_state.doc_summary:
            st.markdown("""
            <div class="card">
                <div class="card-title">📄 Academic Summary</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div class="content-box">{st.session_state.doc_summary}</div>', unsafe_allow_html=True)
        
        # --- STUDY MODES ---
        st.markdown("""
        <div class="card">
            <div class="card-title">🎯 Choose Your Study Mode</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create 4 columns for mode buttons
        mode_col1, mode_col2, mode_col3, mode_col4 = st.columns(4)
        
        with mode_col1:
            if st.button("📘 Learn", key="learn_btn", use_container_width=True):
                st.session_state.mode = "learn"
                # Clear previous content to force regeneration
                st.session_state.explanation_content = None
        
        with mode_col2:
            if st.button("🧠 Key Ideas", key="key_btn", use_container_width=True):
                st.session_state.mode = "key"
                st.session_state.key_points_content = None
        
        with mode_col3:
            if st.button("🎯 Practice", key="practice_btn", use_container_width=True):
                st.session_state.mode = "practice"
                st.session_state.mcqs = None
        
        with mode_col4:
            if st.button("⚡ Exam Cram", key="exam_btn", use_container_width=True):
                st.session_state.mode = "exam"
                st.session_state.exam_cram_content = None
        
        # Show active mode
        if st.session_state.mode:
            mode_names = {
                "learn": "📘 Learn Mode",
                "key": "🧠 Key Ideas Mode", 
                "practice": "🎯 Practice Mode",
                "exam": "⚡ Exam Cram Mode"
            }
            st.markdown(f"""
            <div class="active-mode">
                <strong>Active Mode:</strong> {mode_names[st.session_state.mode]}
            </div>
            """, unsafe_allow_html=True)
        
        # --- SECTION SELECTION ---
        if st.session_state.sections:
            st.markdown("""
            <div class="card">
                <div class="card-title">📚 Select Section to Study</div>
            </div>
            """, unsafe_allow_html=True)
            
            section_titles = [sec["title"] for sec in st.session_state.sections]
            selected_title = st.selectbox(
                "Choose a section:",
                section_titles,
                format_func=lambda x: f"📖 {x}",
                key="section_select"
            )
            
            for sec in st.session_state.sections:
                if sec["title"] == selected_title:
                    st.session_state.selected_section = sec
        
        # --- MAIN CONTENT DISPLAY BASED ON MODE ---
        if st.session_state.selected_section:
            sec = st.session_state.selected_section
            
            # Display section title
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📖 {sec['title']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- LEARN MODE ---
            if st.session_state.mode == "learn":
                if st.session_state.explanation_content is None:
                    with st.spinner("📖 Generating comprehensive explanation..."):
                        explanation = llm.invoke(f"Provide a comprehensive yet simple explanation with examples:\n{sec['content']}").content
                        st.session_state.explanation_content = explanation
                st.markdown(f'<div class="content-box">{st.session_state.explanation_content}</div>', unsafe_allow_html=True)
            
            # --- KEY IDEAS MODE ---
            elif st.session_state.mode == "key":
                if st.session_state.key_points_content is None:
                    with st.spinner("🧠 Extracting key concepts..."):
                        points = llm.invoke(f"Extract 5-7 key points, concepts, and main takeaways:\n{sec['content']}").content
                        st.session_state.key_points_content = points
                st.markdown(f'<div class="content-box">{st.session_state.key_points_content}</div>', unsafe_allow_html=True)
            
            # --- PRACTICE MODE ---
            elif st.session_state.mode == "practice":
                if st.session_state.mcqs is None:
                    with st.spinner("🎯 Generating practice questions..."):
                        mcqs = llm.invoke(f"Generate 5 multiple-choice questions with answers:\n{sec['content']}").content
                        st.session_state.mcqs = mcqs
                
                st.markdown(f'<div class="content-box">{st.session_state.mcqs}</div>', unsafe_allow_html=True)
                
                # Quick switch to exam cram
                if st.button("📝 Generate Exam Cram Notes", key="quick_exam", use_container_width=True):
                    st.session_state.mode = "exam"
                    st.session_state.exam_cram_content = None
                    st.rerun()
            
            # --- EXAM CRAM MODE ---
            elif st.session_state.mode == "exam":
                if st.session_state.exam_cram_content is None:
                    with st.spinner("⚡ Creating revision notes..."):
                        cram = llm.invoke(f"Create concise, bullet-point revision notes with key formulas and concepts:\n{sec['content']}").content
                        st.session_state.exam_cram_content = cram
                
                st.markdown(f'<div class="content-box">{st.session_state.exam_cram_content}</div>', unsafe_allow_html=True)
                
                # PDF Generation
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                
                for line in st.session_state.exam_cram_content.split("\n"):
                    pdf.multi_cell(0, 8, line)
                
                pdf_output = "exam_cram.pdf"
                pdf.output(pdf_output)
                
                with open(pdf_output, "rb") as f:
                    st.download_button(
                        label="📥 Download Revision Notes (PDF)",
                        data=f,
                        file_name="study_notes.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        
        # If no mode selected, show prompt
        elif not st.session_state.mode:
            st.info("👆 Select a study mode above to start learning!")
