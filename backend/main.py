"""
======================================================
Ashish Soni Portfolio — RAG Chatbot Backend
FastAPI + LangChain + ChromaDB + HuggingFace Embeddings
======================================================
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────
app = FastAPI(title="Ashish Soni RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production: set to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global RAG Chain ───────────────────────────────
rag_chain = None
VECTOR_DB_PATH = "./chroma_db"
KNOWLEDGE_DIR = "./knowledge"

# ── Request/Response Models ────────────────────────
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []


# ── RAG System Initialization ──────────────────────
def build_rag_chain():
    """
    Builds the RAG pipeline:
    1. Load documents from /knowledge directory
    2. Split into chunks
    3. Embed with HuggingFace (free, local)
    4. Store in ChromaDB (free, local)
    5. Create retrieval chain with Groq LLM (free tier)
    """
    global rag_chain

    logger.info("Building RAG chain...")

    # 1. Load documents
    loader = DirectoryLoader(KNOWLEDGE_DIR, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks")

    # 3. Embeddings (free, runs locally — no API key needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # 4. Vector store (ChromaDB — local, free)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    vectordb.persist()
    logger.info("Vector store built and persisted")

    # 5. LLM — Groq (free tier, very fast inference)
    # Sign up at: https://console.groq.com — free API key
    llm = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0.3,
        max_tokens=512,
    )

    # 6. Custom RAG Prompt
    prompt_template = """You are an AI assistant for Ashish Soni's portfolio website.
Answer the question based ONLY on the context provided below.
Be concise, professional, and helpful. Speak about Ashish in third person.
If the answer is not in the context, say "I don't have that information, but you can contact Ashish directly at ashishsoni243k@gmail.com"

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 7. RetrievalQA chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    logger.info("RAG chain ready!")
    return rag_chain


# ── Startup Event ──────────────────────────────────
@app.on_event("startup")
async def startup_event():
    build_rag_chain()


# ── Routes ─────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Ashish Soni RAG Chatbot API"}

@app.get("/health")
def health():
    return {"status": "healthy", "rag_ready": rag_chain is not None}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not ready")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = rag_chain({"query": request.query})
        answer = result.get("result", "Sorry, I couldn't find an answer.")
        sources = []

        # Extract source document names for attribution
        source_docs = result.get("source_documents", [])
        for doc in source_docs:
            src = doc.metadata.get("source", "")
            if src and src not in sources:
                sources.append(src.split("/")[-1])  # just filename

        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild-index")
async def rebuild_index():
    """Re-index the knowledge base. Call after updating knowledge files."""
    try:
        build_rag_chain()
        return {"status": "success", "message": "RAG index rebuilt"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
