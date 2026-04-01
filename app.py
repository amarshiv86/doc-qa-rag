import os
import gradio as gr
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from huggingface_hub import InferenceClient

# ── Config ────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "HuggingFaceH4/zephyr-7b-beta"
CHROMA_DIR  = "/tmp/chroma_db"
HF_TOKEN    = os.getenv("HF_TOKEN", "")

# ── Embeddings ────────────────────────────────────────────────
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Embeddings ready ✓")

# ── LLM client ────────────────────────────────────────────────
client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN if HF_TOKEN else None)

# ── State ─────────────────────────────────────────────────────
vectorstore = None
current_doc = None

# ── PDF processing ────────────────────────────────────────────
def process_pdf(pdf_file):
    global vectorstore, current_doc
    if pdf_file is None:
        return "Please upload a PDF file."
    try:
        reader = PdfReader(pdf_file.name)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        if not text.strip():
            return "Could not extract text from this PDF."
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks   = splitter.split_text(text)
        docs     = [Document(page_content=c) for c in chunks]
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
        current_doc = os.path.basename(pdf_file.name)
        return f"✅ Processed **{current_doc}** — {len(reader.pages)} pages, {len(chunks)} chunks indexed. Ready!"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ── RAG query ─────────────────────────────────────────────────
# ── RAG query ─────────────────────────────────────────────────
def answer_question(question, history):
    global vectorstore
    if not question.strip():
        return history, ""
    
    if vectorstore is None:
        # Gradio 5+ uses a list of dicts for history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "⚠️ Please upload a PDF first."})
        return history, ""
    
    try:
        retriever     = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(question)
        context       = "\n\n".join([d.page_content for d in relevant_docs])
        
        # Format the prompt for a Chat model
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant. Answer using ONLY the context provided. If the answer is not there, say 'I couldn't find that in the document.'"
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]

        # Use chat_completion instead of text_generation
        response = client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.3,
            stop=["Question:", "Context:"] # Changed from stop_sequences to stop
        )
        
        answer = response.choices[0].message.content
        
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, ""

    except Exception as e:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return history, ""

def clear_chat():
    return [], ""

def clear_db():
    global vectorstore, current_doc
    vectorstore = None
    current_doc = None
    return "🗑️ Document cleared.", [], ""

# ── UI ────────────────────────────────────────────────────────
with gr.Blocks(title="Document Q&A · RAG") as demo:
    gr.Markdown("""
    # 📄 Document Q&A — RAG Pipeline
    Upload a PDF and ask questions. Powered by **LangChain + ChromaDB + Zephyr-7B**.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input     = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_btn    = gr.Button("→ Process PDF", variant="primary")
            upload_status = gr.Markdown("Upload a PDF to get started.")
            clear_btn     = gr.Button("🗑️ Clear document", variant="secondary")
            gr.Markdown("""
            ### How it works
            1. Upload any PDF document
            2. Text is chunked and embedded into ChromaDB
            3. Your question retrieves the most relevant chunks
            4. Zephyr-7B generates an answer from those chunks

            ### Tips
            - Ask specific questions about the document
            - Works best with text-based PDFs
            - Try: *"What is the main topic?"*
            """)

        with gr.Column(scale=2):
            chatbot  = gr.Chatbot(label="Chat", height=450)
            question = gr.Textbox(
                label="Ask a question about the document",
                placeholder="e.g. What are the main conclusions?",
                lines=2,
            )
            with gr.Row():
                ask_btn        = gr.Button("→ Ask", variant="primary")
                clear_chat_btn = gr.Button("Clear chat")

    upload_btn.click(fn=process_pdf, inputs=pdf_input, outputs=upload_status)
    ask_btn.click(fn=answer_question, inputs=[question, chatbot], outputs=[chatbot, question])
    question.submit(fn=answer_question, inputs=[question, chatbot], outputs=[chatbot, question])
    clear_chat_btn.click(fn=clear_chat, outputs=[chatbot, question])
    clear_btn.click(fn=clear_db, outputs=[upload_status, chatbot, question])

    gr.Markdown("---\nPart of the [AI Engineer Portfolio](https://github.com/amarshiv86)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
