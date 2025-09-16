# GPU-Optimized RAG Chat Interface

A professional Retrieval-Augmented Generation (RAG) chat application built with Streamlit. It supports uploading and querying various document types (PDF, DOCX, MD, TXT, LOG, Excel, CSV) while leveraging GPU acceleration (via ROCm for AMD GPUs or CUDA) for faster processing. The app combines local document knowledge with general AI capabilities using Ollama models.

## Features
- **Document Support**: PDF, Word (DOCX), Markdown (MD), Text/Logs (TXT/LOG), Excel (XLSX/XLS), CSV.
- **GPU Optimization**: Automatic detection and usage of GPU (ROCm/CUDA) for embeddings and processing; falls back to CPU.
- **RAG Pipeline**: Uses FAISS for vector storage, Sentence Transformers for embeddings, and Ollama for LLM inference.
- **Parallel Processing**: Multi-threaded document loading, splitting, and embedding for efficiency.
- **Chat Interface**: Persistent history, professional UI with animations and device status indicators.
- **Structured Data Handling**: Optimized for Excel/CSV with summaries, column info, and JSON storage.
- **Flexible Querying**: Answers from documents or general knowledge; cites sources.

## Requirements
- Python 3.10+ (tested on 3.12)
- Ollama installed and running (for LLM inference: `ollama serve`)
- GPU drivers (optional but recommended): ROCm for AMD or CUDA for NVIDIA.
- Dependencies listed in `requirements.txt`.

## Installation
1. Clone the repository:git clone https://github.com/yourusername/streamlit-rag-demo.git
cd streamlit-rag-demo
2. Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
pip install -r requirements.txt
- Note: For GPU support, ensure PyTorch is installed with ROCm/CUDA (e.g., `pip install torch --index-url https://download.pytorch.org/whl/rocm5.7/` for ROCm).
4. Set up environment variables in `.env` (copy from `.env.example` if available, or use the provided template):
OLLAMA_MODEL=qwen3:0.6b  # Or your preferred Ollama model
CORPUS_DIR=./developer_corpus
FAISS_DIR=./vector_store/faiss_index
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OPENAI_API_KEY=your_openai_key_here  # Optional for OpenAI integrations
5. Pull the Ollama model:
ollama pull qwen3:0.6b  # Replace with your model
## Usage
1. Start Ollama server in a separate terminal:
ollama serve
2. Run the Streamlit app:
streamlit run streamlit_ragdemo.py
3. Open in your browser (usually http://localhost:8501).
4. **Upload Documents**: Use the expander to upload files. Process them to build the index.
5. **Chat**: Ask questions in the input field. The app retrieves from documents if relevant or uses general knowledge.
6. **Sidebar Controls**:
- Select device (GPU/CPU).
- Change Ollama model.
- Reload/clear index.
- View chat history.
### Example Queries
- "Summarize the key points from my uploaded PDF."
- "What is the total rows in the CSV file?"
- "Explain how this app works." (Uses general knowledge.)

## Configuration
- **Device Management**: Auto-detects GPU; force CPU in sidebar.
- **Customization**: Edit `CHUNK_SIZE`, `TOP_K` in `.env` for retrieval tuning.
- **Folders**: Documents go in `CORPUS_DIR`; index in `FAISS_DIR`.

## Performance Tips
- **GPU Acceleration**: 3-10x faster for large docs/embeddings. Ensure ROCm/CUDA is set up.
- **CPU Fallback**: Uses multi-threading for parallel ops.
- **Memory**: Clear GPU cache periodically for long sessions.

## Troubleshooting
- **Ollama Connection**: Ensure `ollama serve` is running.
- **GPU Issues**: Check `torch.cuda.is_available()` in Python console.
- **Embedding Errors**: Verify Sentence Transformers model downloads successfully.
- **File Errors**: Unsupported formats or corrupted files will show warnings.

## Contributing
Pull requests welcome! For major changes, open an issue first.

## License
MIT License (or your chosen license).

---
Built with ❤️ using Streamlit, LangChain, and Ollama. For questions, open an issue.