# 📄 Document Q&A — RAG Pipeline

Ask questions about any PDF document using a **Retrieval-Augmented Generation (RAG)**
pipeline. Upload a PDF, ask a question, and get an answer grounded in the document content.

---

## 🌐 Live Demo

👉 **[Try it live on HF Spaces](https://amarshiv86-doc-qa-rag.hf.space)**

---

## 🏗️ Architecture

```
PDF Upload
    ↓
pypdf — extract text
    ↓
RecursiveCharacterTextSplitter — chunk into 500-token pieces
    ↓
all-MiniLM-L6-v2 — embed chunks into vectors
    ↓
ChromaDB — store + index vectors
    ↓
User Question → embed → similarity search → top-4 chunks
    ↓
Zephyr-7B (HF Inference API) — generate answer from chunks
    ↓
Answer grounded in document
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| PDF parsing | pypdf |
| Text chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local, free) |
| Vector store | ChromaDB |
| LLM | HuggingFaceH4/zephyr-7b-beta (HF Inference API, free) |
| Orchestration | LangChain |
| UI | Gradio |
| CI/CD | GitHub Actions |
| Hosting | Hugging Face Spaces |

---

## 📁 Project Structure

```
doc-qa-rag/
├── app.py              # Gradio app — full RAG pipeline
├── requirements.txt    # Pinned dependencies
├── .gitignore
└── .github/
    └── workflows/
        └── deploy.yml  # Auto-deploy to HF Spaces on push
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/amarshiv86/doc-qa-rag
cd doc-qa-rag
pip install -r requirements.txt
python app.py
# → open http://localhost:7860
```

---

## ⚙️ How RAG Works

**Without RAG:** LLM answers from training data only — can hallucinate or say "I don't know."

**With RAG:**
1. Document is chunked and embedded into a vector store
2. Question is embedded and matched against stored chunks
3. Most relevant chunks are retrieved
4. LLM generates an answer using ONLY those chunks as context
5. Answer is grounded in the actual document — no hallucination

---

## 🔗 Part of AI Engineer Portfolio

| Project | Description |
|---|---|
| P1 — [Weather Prediction](https://github.com/amarshiv86/weather-mlops-pipeline) | Classical ML + MLOps |
| P2 — [Sentiment Analysis](https://github.com/amarshiv86/sentiment-analysis-mlops-pipeline) | Fine-tuned distilBERT + drift detection |
| P3 — [Image Captioning](https://github.com/amarshiv86/image-captioning) | Multi-modal ViT+GPT2 |
| P4 — Document Q&A (this repo) | RAG pipeline with LangChain + ChromaDB |

---

## 📄 License

MIT
