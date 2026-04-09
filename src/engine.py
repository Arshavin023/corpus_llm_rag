import os
import warnings
import time
import numpy as np
from typing import List, Dict
import json
import html
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    UnstructuredHTMLLoader,
    UnstructuredXMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
warnings.filterwarnings(
    "ignore",
    message="Relevance scores must be between 0 and 1"
)
# --- CONFIG ---
DATA_DIR = "src/data"
CHROMA_DIR = "src/chroma_db"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
TOP_K = 5  # increased to improve grounding

# --- Preload embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
# --- Document loaders ---
def load_documents():
    docs = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith((".txt", ".md")):
            loader = TextLoader(path)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".html"):
            loader = UnstructuredHTMLLoader(path)
        elif file.endswith(".xml"):
            loader = UnstructuredXMLLoader(path)
        else:
            continue
        loaded_docs = loader.load()
        # Ensure source metadata is present
        for doc in loaded_docs:
            doc.metadata["source"] = os.path.basename(doc.metadata.get("source", file))
        docs.extend(loaded_docs)
    return docs

def build_index():
    """Build and persist Chroma vectorstore"""
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        # keep_metadata=True  # preserve source for citations
    )
    chunks = splitter.split_documents(documents)
    vectordb_local = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vectordb_local

# --- Preload vectorstore ---
# if os.path.exists(CHROMA_DIR):
#     vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
# else:
#     vectordb = build_index()

_vectordb = None
def get_vectordb():
    global _vectordb
    if _vectordb is None:
        if os.path.exists(CHROMA_DIR):
            _vectordb = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )
        else:
            _vectordb = build_index()
    return _vectordb

# --- Preload LLM ---
def get_llm():
    return ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

# --- RAG functions ---
def generate_answer(query: str, context_docs: List, llm_instance=None) -> Dict:
    """
    Generates an answer using the provided context_docs.
    Always attaches citations with 'source' and snippet.
    """
    # if llm_instance is None:
    #     llm_instance = llm
    if llm_instance is None:
        llm_instance = get_llm()

    if not context_docs:
        return {"answer": "I can only answer questions about company policies.", "citations": []}

    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful policy assistant. Answer based ONLY on the provided context. "
                   "If the answer is not in the context, say 'I can only answer about our policies.' "
                   "Keep the response concise."),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    formatted_prompt = prompt.format_messages(context=context_text, question=query)
    response = llm_instance.invoke(formatted_prompt)

    citations = [
        {"source": doc.metadata.get("source", "unknown"), "snippet": doc.page_content[:200]}
        for doc in context_docs
    ]

    return {"answer": response.content, "citations": citations}

def query_rag_stream(query: str, top_k: int = TOP_K):
    """
    Streaming RAG pipeline with embedding-based reranking:
    1. Retrieve candidate docs
    2. Rerank by embedding similarity to query
    3. Yield citations first
    4. Stream LLM output in JSON-safe chunks
    """
    # 1. DIRECT RETRIEVAL (Replace the candidate_docs and manual loop here)
    # This retrieves docs and their similarity scores in one go
    docs_with_scores = get_vectordb().similarity_search_with_relevance_scores(query, k=top_k)

    # 2. EXTRACT DOCUMENTS
    # We strip the scores and just keep the Document objects for the LLM
    docs = [doc for doc, score in docs_with_scores]

    # FALLBACK: If vector store returns nothing, do a standard search
    if not docs:
        docs = get_vectordb().similarity_search(query, k=2)

    # 3. CONTEXT PREPARATION
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # 4. CITATIONS PREPARATION
    citations = [
        {"source": doc.metadata.get("source", "unknown"), "snippet": doc.page_content[:200]}
        for doc in docs
    ]

    # 6. Prompt for LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict policy assistant. Answer ONLY based on context. "
                   "If not found, say you don't know. Be concise."),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    # chain = prompt | llm
    chain = prompt | get_llm()

    # 7. Yield citations first
    yield (json.dumps({"citations": citations, "content": ""}) + "\n").encode("utf-8")

    # 8. Stream LLM output safely
    for text_chunk in chain.stream({"context": context_text, "question": query}):
        if hasattr(text_chunk, "content") and text_chunk.content:
            yield (json.dumps({"content": text_chunk.content}) + "\n").encode("utf-8")
            # safe_content = html.escape(text_chunk.content)  # escape quotes, newlines
            # yield (json.dumps({"content": safe_content}) + "\n").encode("utf-8")

if __name__ == "__main__":
    print("Engine ready. Chroma vectorstore and LLM preloaded.")