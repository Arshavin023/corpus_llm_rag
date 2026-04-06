import os
import time
from typing import List, Dict

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma

# -------------------
# CONFIG
# -------------------
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4

# -------------------
# LOAD DOCUMENTS
# -------------------
def load_documents():
    docs = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)

        if file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(path)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".html"):
            loader = UnstructuredHTMLLoader(path)
        else:
            continue

        docs.extend(loader.load())

    return docs


# -------------------
# SPLIT DOCUMENTS
# -------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


# -------------------
# EMBEDDINGS + VECTOR STORE
# -------------------
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    return vectordb


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )


# -------------------
# SIMPLE GENERATION (LLM MOCK OR API)
# -------------------
def generate_answer(query: str, context_docs: List) -> Dict:
    """
    Replace with Groq/OpenRouter if needed.
    This is a simple grounded answer builder.
    """

    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    if not context_docs:
        return {
            "answer": "I can only answer questions about company policies.",
            "citations": []
        }

    # Simple grounded response
    answer = f"""
Based on company policies:

{context_docs[0].page_content[:500]}...

"""

    citations = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "snippet": doc.page_content[:200]
        }
        for doc in context_docs
    ]

    return {
        "answer": answer.strip(),
        "citations": citations
    }


# -------------------
# MAIN QUERY FUNCTION
# -------------------
def query_rag(query: str):
    start = time.time()

    vectordb = load_vectorstore()
    docs = vectordb.similarity_search(query, k=TOP_K)

    result = generate_answer(query, docs)

    latency = time.time() - start

    return {
        "query": query,
        "answer": result["answer"],
        "citations": result["citations"],
        "latency": latency
    }


# -------------------
# INITIAL INDEX BUILD
# -------------------
def build_index():
    docs = load_documents()
    chunks = split_documents(docs)
    create_vectorstore(chunks)


if __name__ == "__main__":
    print("Building index...")
    build_index()
    print("Done.")