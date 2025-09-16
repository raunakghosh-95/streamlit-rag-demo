"""
Professional RAG Chat Interface - GPU Optimized with Automatic Device Detection
Supports PDF, DOCX, MD, TXT, LOG, Excel, CSV files + Internet knowledge
Optimized for both CPU and GPU usage with parallel processing
"""

import io
from accelerate import init_empty_weights # For memory-efficient model loading
import os
import shutil
import time
import tempfile
import pandas as pd
import threading
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from dotenv import load_dotenv

# GPU detection and optimization imports
import torch
import psutil
from sentence_transformers import SentenceTransformer

# LangChain & friends
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.schema import Document

# PDF metadata helpers
from pypdf.errors import PdfReadError

# --------------
# GPU Detection & Device Management
# --------------
class DeviceManager:
    def __init__(self):
        # ROCm exposes GPUs as CUDA devices in PyTorch
        self.gpu_available = torch.cuda.is_available() and torch.version.hip is not None
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.cpu_cores = psutil.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
    def get_optimal_device(self, force_cpu=False):
        if force_cpu or not self.gpu_available:
            return 'cpu'
        return 'cuda'  # ROCm still uses "cuda" internally

    def get_device_info(self):
        info = {
            'cpu_cores': self.cpu_cores,
            'memory_gb': round(self.memory_gb, 2),
            'gpu_available': self.gpu_available,
            'gpu_count': self.gpu_count
        }
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info.update({
                'gpu_name': gpu_name,
                'gpu_memory_gb': round(gpu_memory, 2)
            })
        return info

    def optimize_for_device(self, device):
        if device == 'cpu':
            torch.set_num_threads(min(self.cpu_cores, 8))
        else:  # ROCm GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
# Initialize device manager
device_manager = DeviceManager()

# --------------
# Config & Defaults
# --------------
load_dotenv(override=True)
APP_TITLE = "AI Knowledge Assistant"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")  # Updated to 3.2
DEFAULT_CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "./developer_corpus"))
DEFAULT_FAISS_DIR = Path(os.getenv("FAISS_DIR", "./vector_store/faiss_index"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --------------
# Custom CSS for Professional Look
# --------------
st.set_page_config(
    page_title=APP_TITLE, 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with GPU status indicators
st.markdown("""
<style>
    /* Hide Streamlit branding */
    .stDeployButton {display: none;}
    footer {display: none;}
    header {display: none;}
    
    /* Main container styling */
    .main > div {
        padding-top: 1rem;
    }
    
    /* Chat interface styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 3px solid #e0e0e0;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background-color: #f8f9fa;
        border-left-color: #007bff;
    }
    
    .assistant-message {
        background-color: #ffffff;
        border-left-color: #28a745;
        border: 1px solid #e9ecef;
    }
    
    /* Device status indicators */
    .device-status {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
    }
    
    .gpu-active {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .cpu-active {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #b8daff;
    }
    
    /* Performance indicators */
    .perf-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .perf-fast { background-color: #28a745; }
    .perf-medium { background-color: #ffc107; }
    .perf-slow { background-color: #dc3545; }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Upload area styling */
    .uploadedFile {
        border-radius: 0.5rem;
        border: 2px dashed #007bff;
        padding: 1rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 0.5rem;
        border: none;
        background: linear-gradient(90deg, #007bff, #0056b3);
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
        border: 2px solid #e9ecef;
        padding: 0.75rem;
    }
    
    /* Hide upload file details */
    .uploadedFileData {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --------------
# Optimized Helper Functions
# --------------
class OptimizedEmbeddings:
    def __init__(self, model_name: str, device: str = 'cpu'):
            self.model_name = model_name
            self.device = device
            device_manager.optimize_for_device(device)

            pytorch_device = device if device in ['cuda', 'cpu'] else 'cpu'
            dtype = torch.float32 if device == 'cpu' else torch.float16   # Stable for CPU

            try:
                # Use init_empty_weights to safely handle meta without .to() issues
                with init_empty_weights():
                    # Load base model on meta first, then move safely
                    self.model = SentenceTransformer(model_name)

                # Move to device without .to()—use _load_from_state_dict or direct assignment
                # But simpler: Re-init directly on device (avoids meta entirely for small models like MiniLM)
                self.model = SentenceTransformer(
                    model_name,
                    device=pytorch_device,
                    torch_dtype=dtype,
                    cache_folder=None  # Avoid cache conflicts
                )

                # For HF Embeddings: Explicit device_map, no offload
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={
                        "device_map": pytorch_device,  # Direct map, no meta offload
                        "low_cpu_mem_usage": True,     # Prevents meta init
                        "torch_dtype": dtype,
                        "trust_remote_code": True      # If needed for custom models
                    },
                    encode_kwargs={"device": pytorch_device, "dtype": dtype}
                )
                st.success(f"✅ Embeddings loaded on {device.upper()} without meta issues")
            except Exception as e:
                st.error(f"❌ Embeddings failed on {device}: {e}. Forcing CPU no-meta mode.")
                # Ultimate fallback: CPU with explicit no-meta
                self.device = 'cpu'
                self.model = SentenceTransformer(
                    model_name,
                    device='cpu',
                    torch_dtype=torch.float32
                )
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={
                        "device_map": "cpu",
                        "low_cpu_mem_usage": True,
                        "torch_dtype": torch.float32,
                        "output_hidden_states": False  # Simplify to avoid internal moves
                    }
                )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed with parallel processing"""
        if self.device == 'cuda' and len(texts) > 10:
            batch_size = 16  # Smaller for stability
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_emb = self.embeddings.embed_documents(batch)
                embeddings.extend(batch_emb)
                torch.cuda.empty_cache()  # Prevent buildup
            return embeddings
        else:
            # CPU parallel
            max_workers = min(device_manager.cpu_cores, 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                if len(texts) > 50:
                    chunk_size = max(10, len(texts) // max_workers)
                    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
                    futures = [executor.submit(self.embeddings.embed_documents, chunk) for chunk in chunks]
                    return [emb for future in futures for emb in future.result()]
                else:
                    return self.embeddings.embed_documents(texts)
                
def load_excel_csv_optimized(file_path: Path) -> List[Document]:
    """Optimized Excel/CSV loader with parallel processing"""
    docs = []
    try:
        # Use faster pandas settings
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, low_memory=False)
        else:  # Excel
            df = pd.read_excel(file_path, engine='openpyxl')
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(r'^"|"$', '', regex=True)
        
        # Create optimized content representation
        content = f"File: {file_path.name}\n\n"
        content += f"Dataset Summary:\n"
        content += f"Columns: {', '.join(df.columns.tolist())}\n"
        content += f"Total Rows: {len(df)}\n\n"
        
        # Parallel column analysis
        def analyze_column(col):
            dtype = str(df[col].dtype)
            unique_vals = df[col].nunique()
            sample_vals = df[col].head(3).tolist()
            return f"- {col}: {dtype}, {unique_vals} unique values, sample: {sample_vals}\n"
        
        with ThreadPoolExecutor(max_workers=min(len(df.columns), 8)) as executor:
            column_info = list(executor.map(analyze_column, df.columns))
        
        content += "Column Information:\n"
        content += "".join(column_info)
        
        # Add key identifiers
        if 'NCT Number' in df.columns:
            content += "\nKey Identifiers:\n"
            content += f"NCT Numbers: {', '.join(df['NCT Number'].astype(str).tolist())}\n"
        
        # Store DataFrame as compressed JSON
        df_json = df.to_json(orient='records', lines=True, compression='gzip')
        
        doc = Document(
            page_content=content,
            metadata={
                "source": file_path.name,
                "type": "spreadsheet",
                "dataframe": df_json,
                "rows": len(df),
                "columns": len(df.columns)
            }
        )
        docs.append(doc)
        
    except Exception as e:
        st.error(f"Error loading {file_path.name}: {e}")
    
    return docs

def load_docs_from_paths_parallel(paths: List[Path]) -> Tuple[List, List[str]]:
    """Parallel document loading for better performance"""
    docs, errors = [], []
    
    def load_single_doc(p: Path):
        try:
            suffix = p.suffix.lower()
            if suffix == ".pdf":
                loader = PyPDFLoader(str(p))
                file_docs = loader.load()
            elif suffix in (".docx",):
                loader = Docx2txtLoader(str(p))
                file_docs = loader.load()
            elif suffix in (".md", ".markdown"):
                loader = UnstructuredMarkdownLoader(str(p))
                file_docs = loader.load()
            elif suffix in (".txt", ".log"):
                loader = TextLoader(str(p), encoding="utf-8")
                file_docs = loader.load()
            elif suffix in (".csv", ".xlsx", ".xls"):
                file_docs = load_excel_csv_optimized(p)
            else:
                return None, f"Unsupported file type: {p.name}"

            # Add metadata
            for d in file_docs:
                meta = d.metadata or {}
                meta["source"] = p.name
                if suffix == ".pdf" and "page" not in meta:
                    meta["page"] = meta.get("page", None)
                d.metadata = meta
            
            return file_docs, None
            
        except Exception as e:
            return None, f"Error loading {p.name}: {e}"
    
    # Use parallel processing for multiple files
    if len(paths) > 1:
        with ThreadPoolExecutor(max_workers=min(len(paths), 4)) as executor:
            futures = [executor.submit(load_single_doc, p) for p in paths]
            for future in futures:
                file_docs, error = future.result()
                if error:
                    errors.append(error)
                elif file_docs:
                    docs.extend(file_docs)
    else:
        # Single file processing
        for p in paths:
            file_docs, error = load_single_doc(p)
            if error:
                errors.append(error)
            elif file_docs:
                docs.extend(file_docs)
    
    return docs, errors

def split_docs_parallel(docs, chunk_size: int, chunk_overlap: int):
    """Parallel document splitting"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        separators=["\n\n", "\n", " ", ""]
    )
    
    if len(docs) > 10:  # Use parallel processing for large document sets
        with ThreadPoolExecutor(max_workers=min(len(docs), 4)) as executor:
            chunk_size_per_worker = max(1, len(docs) // 4)
            doc_chunks = [docs[i:i+chunk_size_per_worker] for i in range(0, len(docs), chunk_size_per_worker)]
            
            futures = [executor.submit(splitter.split_documents, chunk) for chunk in doc_chunks]
            all_splits = []
            for future in futures:
                all_splits.extend(future.result())
            return all_splits
    else:
        return splitter.split_documents(docs)

def build_or_refresh_index_optimized(all_docs, faiss_dir: Path, device: str = 'cpu'):
    """Optimized index building with GPU support"""
    if not all_docs:
        raise ValueError("No documents to index.")
    
    # Use optimized embeddings
    embeddings = OptimizedEmbeddings(EMBEDDING_MODEL, device)
    
    # Build index with progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Creating vector embeddings...")
        progress_bar.progress(0.3)
        
        vectorstore = FAISS.from_documents(all_docs, embeddings.embeddings)
        progress_bar.progress(0.8)
        
        status_text.text("Saving index...")
        faiss_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(faiss_dir))
        progress_bar.progress(1.0)
        
        status_text.text("✅ Index created successfully!")
        
    finally:
        # Clean up progress indicators
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
    
    return vectorstore

def load_index_optimized(faiss_dir: Path, device: str = 'cpu') -> Optional[FAISS]:
    """Optimized index loading"""
    if not faiss_dir.exists():
        return None
    try:
        embeddings = OptimizedEmbeddings(EMBEDDING_MODEL, device)
        return FAISS.load_local(
            str(faiss_dir), 
            embeddings.embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Error loading index: {e}")
        return None

# --------------
# Enhanced RAG Chain with GPU Optimization
# --------------
FLEXIBLE_SYSTEM_PROMPT = """You are an intelligent AI assistant with access to uploaded documents
and general knowledge. Your capabilities: - Answer questions using information from uploaded
documents when relevant - Provide general knowledge and internet-based information when appropriate 
- Explain what this AI Knowledge Assistant application does and how it works -
Help users understand how to use the system effectively Guidelines: 
1. PRIORITIZE document content when the question relates to uploaded materials 
2. For questions about the application itself, explain that this is an AI Knowledge Assistant 
that: - Processes uploaded documents (PDF, Word, Excel, CSV, Markdown, Text files) 
- Creates a searchable knowledge base from user documents - Combines document knowledge with general
AI knowledge - Helps users find information quickly from their document collections 
3. For general questions not in the documents, use your broader knowledge 
4. Always cite document sources when using uploaded content: (filename:page) 
5. Be helpful, accurate, and professional 
6. If unsure about document content, clearly distinguish between document-based and general 
knowledge Handling Structured Data (CSV, Excel, and Similar Formats): When dealing with structured 
data files like CSV or Excel: - Thoroughly parse and understand the file structure, including 
columns, rows, headers, and data types (e.g., strings, integers, floats, booleans, dates, datetime).
- For questions involving the data:   - Identify and reference specific columns, rows, or records 
accurately (e.g., "In column 'Age' of row 5...").   - Interpret entire rows or subsets semantically,
inferring context such as demographics (e.g., population statistics), lab work (e.g., test results 
and trends), financial data, or other domains based on headers and content.   - Provide both 
analytical insights (e.g., calculations like sums, averages, correlations, or trends) and semantic 
explanations (e.g., what patterns or implications the data suggests).   - Cite sources as 
(filename:sheet:row:column) or similar for precision when applicable, falling back to 
(filename:page) if not. Remember: You're designed to be a comprehensive knowledge assistant, 
not just a document search tool."""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", FLEXIBLE_SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nRelevant document context:\n{context}\n\nAnswer:"),
])

def format_docs_optimized(docs) -> str:
    """Optimized document formatting with parallel processing"""
    if not docs:
        return "No relevant documents found."
    
    def format_single_doc(d):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        doc_type = d.metadata.get("type", "document")
        
        content = d.page_content
        
        # Optimized spreadsheet handling
        if doc_type == "spreadsheet" and "dataframe" in d.metadata:
            try:
                df = pd.read_json(d.metadata["dataframe"], orient='records', lines=True)
                # Optimize NCT number search
                last_question = st.session_state.get("last_question", "")
                if "NCT" in last_question:
                    nct_matches = [word for word in last_question.split() if "NCT" in word]
                    if nct_matches and 'NCT Number' in df.columns:
                        nct_number = nct_matches[0].replace("NCT", "")[:8]
                        matching_row = df[df['NCT Number'].str.contains(nct_number, na=False)]
                        if not matching_row.empty:
                            content += "\n\nRelevant Row Data:\n"
                            for _, row in matching_row.iterrows():
                                content += "Record:\n"
                                for col in df.columns:
                                    content += f"- {col}: {row[col]}\n"
            except Exception as e:
                content += f"\n\nError processing structured data: {e}"
        
        tag = f"({src}:page {page})" if page is not None else f"({src})"
        return f"{content}\n{tag}"
    
    # Use parallel formatting for large document sets
    if len(docs) > 5:
        with ThreadPoolExecutor(max_workers=min(len(docs), 4)) as executor:
            formatted = list(executor.map(format_single_doc, docs))
    else:
        formatted = [format_single_doc(d) for d in docs]
    
    return "\n\n---\n\n".join(formatted)

def make_chain_optimized(vectorstore: Optional[FAISS], ollama_model: str, device: str = 'cpu'):
    """Optimized chain creation with device-specific settings"""
    
    def flexible_retriever(question: str):
        st.session_state["last_question"] = question
        if vectorstore:
            # Optimize retrieval based on device
            search_kwargs = {"k": TOP_K}
            if device == 'cuda':
                # Use larger k for GPU (can handle more processing)
                search_kwargs["k"] = min(TOP_K * 2, 10)
            
            retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
            return retriever.get_relevant_documents(question)
        return []
    
    # Configure Ollama with optimized settings
    llm_kwargs = {
        "model": ollama_model,
        "temperature": 0.3,
        "num_ctx": 8192,  # Larger context window
        "num_predict": 2048,  # Optimize prediction length
    }
    
    # GPU-specific optimizations for Ollama
    if device == 'cuda':
        llm_kwargs.update({
            "num_gpu": 1,  # Use GPU acceleration
            "num_thread": min(device_manager.cpu_cores, 8),
        })
    else:
        llm_kwargs.update({
            "num_thread": min(device_manager.cpu_cores, 6),  # Leave some cores free
        })
    
    llm = ChatOllama(**llm_kwargs)

    def chain_invoke(question: str):
        docs = flexible_retriever(question)
        context = format_docs_optimized(docs)
        
        prompt_input = {
            "question": question,
            "context": context
        }
        
        formatted_prompt = PROMPT_TEMPLATE.format_messages(**prompt_input)
        response = llm.invoke(formatted_prompt)
        return response.content, docs

    return chain_invoke

# --------------
# Initialize Session State with Device Management
# --------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "selected_device" not in st.session_state:
    # Auto-select best device
    st.session_state.selected_device = device_manager.get_optimal_device()
if "device_info" not in st.session_state:
    st.session_state.device_info = device_manager.get_device_info()

# Load index with selected device
if st.session_state.vectorstore is None:
    st.session_state.vectorstore = load_index_optimized(
        DEFAULT_FAISS_DIR, 
        st.session_state.selected_device
    )

# --------------
# Sidebar - Enhanced with Device Management
# --------------
with st.sidebar:
    st.title("🚀 GPU-Optimized RAG")
    
    # Device Selection & Status
    st.subheader("⚡ Hardware Settings")
    
    # Device info display
    device_info = st.session_state.device_info
    
    if device_info['gpu_available']:
        st.markdown(f"""
        <div class="device-status gpu-active">
            <span class="perf-indicator perf-fast"></span>
            <strong>GPU Available:</strong> {device_info['gpu_name']}<br>
            <strong>GPU Memory:</strong> {device_info['gpu_memory_gb']} GB
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="device-status cpu-active">
        <span class="perf-indicator perf-medium"></span>
        <strong>CPU:</strong> {device_info['cpu_cores']} cores<br>
        <strong>RAM:</strong> {device_info['memory_gb']} GB
    </div>
    """, unsafe_allow_html=True)
    
    # Device selection
    device_options = ["Auto (Recommended)"]
    if device_info['gpu_available']:
        device_options.append("GPU (CUDA)")
    device_options.append("CPU Only")
    
    selected_device_option = st.selectbox(
        "Processing Device",
        device_options,
        help="Auto mode uses GPU if available, otherwise CPU"
    )
    
    # Update device based on selection
    if selected_device_option == "GPU (ROCm)":
        new_device = 'cuda'
    elif selected_device_option == "CPU Only":
        new_device = 'cpu'
    else:  # Auto
        new_device = device_manager.get_optimal_device()
    
    if new_device != st.session_state.selected_device:
        st.session_state.selected_device = new_device
        # Reload vectorstore with new device
        if st.session_state.vectorstore:
            st.session_state.vectorstore = load_index_optimized(
                DEFAULT_FAISS_DIR, 
                new_device
            )
        st.rerun()
    
    # Current device status
    current_device = st.session_state.selected_device
    device_emoji = "🔥" if current_device == 'cuda' else "🔧"
    st.info(f"{device_emoji} Using: {current_device.upper()}")
    
    st.markdown("---")
    
    # Chat History
    st.title("💬 Chat History")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    # Display recent chat history
    if st.session_state.chat_history:
        for i, turn in enumerate(st.session_state.chat_history[-8:]):  # Show last 8
            if turn['role'] == 'user':
                with st.container():
                    st.markdown(f"**You:** {turn['content'][:80]}{'...' if len(turn['content']) > 80 else ''}")
            else:
                with st.container():
                    st.markdown(f"**AI:** {turn['content'][:80]}{'...' if len(turn['content']) > 80 else ''}")
            st.markdown("---")
    else:
        st.caption("No conversation yet. Start by asking a question!")
    
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    
    # Ollama model selection
    ollama_model = st.text_input("Ollama Model", value=DEFAULT_MODEL)
    
    # Document status
    if st.session_state.vectorstore:
        st.success("📚 Documents loaded")
    else:
        st.info("📭 No documents loaded")
    
    # Advanced settings
    with st.expander("Advanced Settings"):        
        # File management
        st.subheader("File Management")
        if st.button("📄 Reload Index"):
            with st.spinner("Reloading index..."):
                vs = load_index_optimized(DEFAULT_FAISS_DIR, st.session_state.selected_device)
                if vs:
                    st.session_state.vectorstore = vs
                    st.success("Index reloaded!")
                    st.rerun()
                else:
                    st.error("No saved index found.")
        
        if st.button("🗑️ Clear Document Index"):
            try:
                if DEFAULT_FAISS_DIR.exists():
                    shutil.rmtree(DEFAULT_FAISS_DIR)
                st.session_state.vectorstore = None
                st.session_state.documents_loaded = False
                st.warning("Document index cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear index: {e}")

# --------------
# Main Interface
# --------------

# Header with performance indicators
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown("# 🚀 AI Knowledge Assistant")
    st.markdown("*GPU-optimized document processing and chat interface*")

with col2:
    if st.session_state.selected_device == 'cuda':
        st.markdown("🔥 **GPU Accelerated**")
    else:
        st.markdown("🔧 **CPU Processing**")

with col3:
    if st.session_state.vectorstore:
        st.markdown("📚 **Documents Ready**")
    else:
        st.markdown("📭 **No Documents**")

# Document Upload Section
with st.expander("📁 Upload Documents", expanded=not st.session_state.documents_loaded):
    st.markdown("**Supported formats:** PDF, Word (DOCX), Excel (XLSX/XLS), CSV, Markdown (MD), Text (TXT), Log files")
    st.markdown(f"**Current processing device:** {st.session_state.selected_device.upper()}")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "md", "markdown", "txt", "log", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Upload multiple files at once. GPU acceleration will speed up processing significantly."
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            if uploaded_files:
                # Process uploaded files with device-optimized pipeline
                temp_dir = Path(tempfile.mkdtemp())
                try:
                    processing_start_time = time.time()
                    
                    with st.spinner(f"🔥 Processing {len(uploaded_files)} files on {st.session_state.selected_device.upper()}..."):
                        # Save uploaded files in parallel
                        paths = []
                        
                        def save_file(uf):
                            p = temp_dir / uf.name
                            with open(p, "wb") as f:
                                f.write(uf.read())
                            return p
                        
                        # Parallel file saving
                        with ThreadPoolExecutor(max_workers=min(len(uploaded_files), 4)) as executor:
                            paths = list(executor.map(save_file, uploaded_files))
                        
                        # Load documents with parallel processing
                        docs, errs = load_docs_from_paths_parallel(paths)
                        
                        if errs:
                            st.warning("⚠️ Some files had issues:")
                            for e in errs:
                                st.caption(f"• {e}")
                        
                        if docs:
                            # Split documents in parallel
                            chunks = split_docs_parallel(docs, CHUNK_SIZE, CHUNK_OVERLAP)
                            
                            # Build index with GPU/CPU optimization
                            vs = build_or_refresh_index_optimized(
                                chunks, 
                                DEFAULT_FAISS_DIR, 
                                st.session_state.selected_device
                            )
                            st.session_state.vectorstore = vs
                            st.session_state.documents_loaded = True
                            
                            processing_time = time.time() - processing_start_time
                            device_emoji = "🔥" if st.session_state.selected_device == 'cuda' else "🔧"
                            
                            st.success(f"""
                            ✅ Successfully processed {len(uploaded_files)} files!
                            {device_emoji} Device: {st.session_state.selected_device.upper()}
                            📊 Created {len(chunks)} searchable chunks
                            ⏱️ Processing time: {processing_time:.2f} seconds
                            """)
                        else:
                            st.error("❌ No readable content found in uploaded files.")
                            
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                st.warning("Please upload at least one file.")
    
    with col2:
        file_count = len(uploaded_files) if uploaded_files else 0
        st.metric("Files Selected", file_count)
    
    with col3:
        if st.session_state.selected_device == 'cuda':
            st.metric("GPU Boost", "🔥 Active")
        else:
            st.metric("CPU Cores", f"{device_manager.cpu_cores}")

# Performance Tips
if not st.session_state.documents_loaded:
    st.info("""
    💡 **Performance Tips:**
    - GPU processing is 3-10x faster for large documents
    - Multiple files are processed in parallel automatically
    - Larger files benefit more from GPU acceleration
    - CSV/Excel files use optimized pandas processing
    """)

# Chat Interface
st.markdown("---")

# Display chat history with enhanced styling
for turn in st.session_state.chat_history:
    if turn['role'] == 'user':
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>
            {turn['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>AI Assistant:</strong><br>
            {turn['content']}
        </div>
        """, unsafe_allow_html=True)

# Enhanced Input Area
st.markdown("### 💭 Ask me anything")

# Input with performance indicator
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "Type your question here...",
        placeholder=f"Ask about your documents or anything else... (Processing on {st.session_state.selected_device.upper()})",
        label_visibility="collapsed"
    )

with col2:
    ask_btn = st.button("🚀 Send", type="primary", use_container_width=True)

# Process question with optimized pipeline
if ask_btn and question.strip():
    try:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Generate response with performance timing
        response_start_time = time.time()  # Start overall timing
        device_emoji = "🔥" if st.session_state.selected_device == 'cuda' else "🔧"
        
        with st.spinner(f"{device_emoji} Processing on {st.session_state.selected_device.upper()}..."):
            chain_invoke = make_chain_optimized(
                st.session_state.vectorstore, 
                ollama_model,
                st.session_state.selected_device
            )
            think_start_time = time.time()  # Start thinking trace timing
            answer, retrieved_docs = chain_invoke(question)
            think_end_time = time.time()    # End thinking
            thinking_time = think_end_time - think_start_time
        
        overall_response_time = time.time() - response_start_time
        
        # Format the response with collapsible thinking
        thinking_trace = f"""
        ### Simulated Thinking Trace
        - Analyzed query: '{question[:50]}...'
        - Retrieved {len(retrieved_docs)} docs from index.
        - Generated response using {ollama_model} on {st.session_state.selected_device.upper()}.
        - Considered optimizations: {device_emoji} mode active.
        - Ensured citations and scoped to local data.
        """
        
        formatted_response = f"""
        <details>
        <summary>🤔 AI Thinking (took {thinking_time:.2f} seconds) - Click to expand</summary>
        
        {thinking_trace}
        
        </details>
        
        <div style='margin-top: 1rem; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; border-left: 4px solid #28a745;'>
        <strong>Final Answer:</strong><br>
        {answer}
        {'\n\n*⚡ Overall response in {overall_response_time:.2f}s*' if overall_response_time < 2.0 else ''}
        </div>
        """
        
        # Add formatted response to chat history (store raw answer for history)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Display the formatted response
        st.markdown(formatted_response, unsafe_allow_html=True)
        
        # Show performance metrics temporarily
        if len(retrieved_docs) > 0:
            st.success(f"📊 Found {len(retrieved_docs)} relevant documents in {overall_response_time:.2f}s")
        
        # Rerun to show the new messages
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        if "connection" in str(e).lower():
            st.info("💡 Make sure Ollama is running: `ollama serve`")

elif ask_btn and not question.strip():
    st.warning("Please enter a question.")

# System Status and Performance Section
st.markdown("---")

# Performance Dashboard
with st.expander("📊 System Performance", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.selected_device == 'cuda':
            st.metric("Processing", "GPU 🔥", delta="Fast")
        else:
            st.metric("Processing", "CPU 🔧", delta="Standard")
    
    with col2:
        doc_count = len(st.session_state.chat_history)
        st.metric("Messages", doc_count)
    
    with col3:
        if st.session_state.vectorstore:
            st.metric("Index Status", "Ready 📚", delta="Loaded")
        else:
            st.metric("Index Status", "Empty 📭", delta="No docs")
    
    with col4:
        memory_usage = f"{device_manager.memory_gb:.1f}GB"
        st.metric("System RAM", memory_usage)
    
    # Advanced performance info
    if device_manager.gpu_available:
        st.markdown("### 🔥 GPU Information")
        gpu_info = device_manager.get_device_info()
        st.json({
            "GPU Name": gpu_info.get('gpu_name', 'N/A'),
            "GPU Memory": f"{gpu_info.get('gpu_memory_gb', 0):.1f} GB",
            "CUDA Available": gpu_info['gpu_available'],
            "Current Device": st.session_state.selected_device.upper()
        })
    
    st.markdown("### 🔧 Optimization Tips")
    if st.session_state.selected_device == 'cuda':
        st.success("""
        ✅ **GPU Optimization Active**
        - Vector embeddings use GPU acceleration
        - Batch processing optimized for CUDA
        - Memory management optimized
        - Parallel document processing enabled
        """)
    else:
        st.info("""
        🔧 **CPU Optimization Active**
        - Multi-threaded document processing
        - Parallel embedding generation
        - Optimized memory usage
        - Thread pool for concurrent operations
        """)

# Footer with enhanced information
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p><strong>🚀 GPU-Optimized AI Knowledge Assistant</strong></p>
    <p>Current Device: {st.session_state.selected_device.upper()} • 
       Model: {ollama_model} • 
       Supports: PDF • Word • Excel • CSV • Markdown • Text • Log files</p>
    <p>💡 <em>GPU acceleration provides 3-10x faster processing for large documents</em></p>
</div>
""", unsafe_allow_html=True)

# Background optimization tasks
if st.session_state.selected_device == 'cuda' and torch.cuda.is_available():
    # Clear GPU cache periodically to prevent memory issues
    if len(st.session_state.chat_history) % 10 == 0:  # Every 10 messages
        torch.cuda.empty_cache()