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
    
    /* Section Cards */
    .section-card {
        background: #f7fafc;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.2s;
        border: 2px solid transparent;
    }
    
    .section-card:hover {
        border-color: #667eea;
        background: white;
    }
    
    .section-card.selected {
        border-color: #667eea;
        background: white;
        box-shadow: 0 2px 8px rgba(102,126,234,0.2);
    }
    
    /* Mode Buttons */
    .mode-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        margin: 0.5rem;
        border: none;
    }
    
    .mode-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    
    /* Content Styles */
    .content-box {
        background: #f7fafc;
        border-radius: 12px;
        padding: 1.5rem;
        line-height: 1.6;
        color: #2d3748;
    }
    
    /* Progress Indicator */
    .progress-step {
        display: inline-block;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: #e2e8f0;
        text-align: center;
        line-height: 30px;
        margin: 0 0.5rem;
        font-weight: 600;
    }
    
    .progress-step.active {
        background: #667eea;
        color: white;
    }
    
    /* Download Button */
    .download-btn {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        text-decoration: none;
        display: inline-block;
        transition: all 0.2s;
    }
    
    .download-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(72,187,120,0.3);
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
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
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
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
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

# --- MAIN LAYOUT ---
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

# --- STUDY CONTENT SECTION ---
if st.session_state.upload_complete:
    # Two-column layout for study content
    col_left, col_right = st.columns([1.2, 2.8])
    
    with col_left:
        # --- SUMMARY CARD ---
        if st.session_state.doc_summary:
            st.markdown("""
            <div class="card">
                <div class="card-title">📄 Executive Summary</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div class="content-box">{st.session_state.doc_summary}</div>', unsafe_allow_html=True)
        
        # --- STUDY MODES ---
        st.markdown("""
        <div class="card">
            <div class="card-title">🎯 Study Mode</div>
        </div>
        """, unsafe_allow_html=True)
        
        mode_options = {
            "learn": {"icon": "📘", "label": "Learn", "desc": "Detailed explanations"},
            "key": {"icon": "🧠", "label": "Key Ideas", "desc": "Essential concepts"},
            "practice": {"icon": "🎯", "label": "Practice", "desc": "Test yourself"},
            "exam": {"icon": "⚡", "label": "Exam Cram", "desc": "Quick revision"}
        }
        
        for mode_key, mode_info in mode_options.items():
            if st.button(f"{mode_info['icon']} {mode_info['label']}", key=f"mode_{mode_key}", help=mode_info['desc']):
                st.session_state.mode = mode_key
                if mode_key == "practice":
                    st.session_state.mcqs = None
        
        if st.session_state.mode:
            st.markdown(f"""
            <div style="background: #e6f7ff; padding: 0.5rem; border-radius: 8px; margin-top: 1rem; text-align: center;">
                Active Mode: <strong>{mode_options[st.session_state.mode]['label']}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    with col_right:
        # --- SECTION SELECTION ---
        if st.session_state.sections:
            st.markdown("""
            <div class="card">
                <div class="card-title">📚 Select Topic</div>
            </div>
            """, unsafe_allow_html=True)
            
            section_titles = [sec["title"] for sec in st.session_state.sections]
            selected_title = st.selectbox(
                "Choose a section to study:",
                section_titles,
                format_func=lambda x: f"📖 {x}"
            )
            
            for sec in st.session_state.sections:
                if sec["title"] == selected_title:
                    st.session_state.selected_section = sec
        
        # --- MAIN CONTENT DISPLAY ---
        if st.session_state.selected_section:
            sec = st.session_state.selected_section
            
            st.markdown(f"""
            <div class="card">
                <div class="card-title">{sec['title']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- LEARN MODE ---
            if st.session_state.mode == "learn":
                with st.spinner("Analyzing and explaining..."):
                    explanation = llm.invoke(f"Provide a comprehensive yet simple explanation:\n{sec['content']}").content
                st.markdown(f'<div class="content-box">{explanation}</div>', unsafe_allow_html=True)
            
            # --- KEY IDEAS MODE ---
            elif st.session_state.mode == "key":
                with st.spinner("Extracting key concepts..."):
                    points = llm.invoke(f"Extract 5-7 key points, concepts, and takeaways:\n{sec['content']}").content
                st.markdown(f'<div class="content-box">{points}</div>', unsafe_allow_html=True)
            
            # --- PRACTICE MODE ---
            elif st.session_state.mode == "practice":
                if st.session_state.mcqs is None:
                    with st.spinner("Generating practice questions..."):
                        mcqs = llm.invoke(f"Generate 5 multiple-choice questions with answers:\n{sec['content']}").content
                        st.session_state.mcqs = mcqs
                
                st.markdown(f'<div class="content-box">{st.session_state.mcqs}</div>', unsafe_allow_html=True)
                
                if st.button("📝 Generate Exam Cram Notes", key="gen_exam"):
                    st.session_state.mode = "exam"
                    st.rerun()
            
            # --- EXAM CRAM MODE ---
            elif st.session_state.mode == "exam":
                with st.spinner("Creating revision notes..."):
                    cram = llm.invoke(f"Create concise, bullet-point revision notes with key formulas and concepts:\n{sec['content']}").content
                
                st.markdown(f'<div class="content-box">{cram}</div>', unsafe_allow_html=True)
                
                # PDF Generation
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                
                for line in cram.split("\n"):
                    pdf.multi_cell(0, 8, line)
                
                pdf.output("exam_cram.pdf")
                
                with open("exam_cram.pdf", "rb") as f:
                    st.download_button(
                        label="📥 Download Revision Notes (PDF)",
                        data=f,
                        file_name="study_notes.pdf",
                        mime="application/pdf"
                    )
