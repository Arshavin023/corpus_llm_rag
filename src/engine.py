import os
import time
from typing import List, Dict
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    UnstructuredHTMLLoader,
    UnstructuredXMLLoader  # Add this import
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- CONFIG ---
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4

# Initialize embeddings once for reuse
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_documents():
    docs = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        # Determine the correct loader based on extension
        if file.endswith((".txt", ".md")):
            loader = TextLoader(path)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".html"):
            loader = UnstructuredHTMLLoader(path)
        elif file.endswith(".xml"):
            # This will extract the text from the XML tags
            loader = UnstructuredXMLLoader(path)
        else:
            print(f"Skipping unsupported file: {file}")
            continue
        docs.extend(loader.load())
    
    return docs

def build_index():
    """Step 2: Ingestion and Indexing"""
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    
    # Store vectors in local Chroma DB
    vectordb = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_DIR
    )
    return vectordb

def generate_answer(query: str, context_docs: List) -> Dict:
    """Step 3: Retrieval and Generation (RAG)"""
    if not context_docs:
        return {
            "answer": "I can only answer questions about company policies.",
            "citations": []
        }

    # Setup LLM (Using Groq as per project suggestions)
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.1-8b-instant", 
        api_key=os.getenv("GROQ_API_KEY") 
    )

    # Building the context string
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    # System Prompt with Guardrails
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful policy assistant. Answer based ONLY on the provided context. "
                   "If the answer is not in the context, say 'I can only answer about our policies.' "
                   "Always keep the response concise."),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])

    # chain = prompt | llm
    # response = chain.invoke({"context": context_text, "question": query})
    formatted_prompt = prompt.format_messages(
        context=context_text,
        question=query
        )

    response = llm.invoke(formatted_prompt)

    citations = [
        {"source": doc.metadata.get("source", "unknown"), "snippet": doc.page_content[:200]}
        for doc in context_docs
    ]

    return {"answer": response.content, "citations": citations}

def query_rag(query: str):
    start = time.time()
    
    # Load existing vectorstore
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    docs = vectordb.similarity_search(query, k=TOP_K)

    result = generate_answer(query, docs)
    latency = time.time() - start

    return {
        "query": query,
        "answer": result["answer"],
        "citations": result["citations"],
        "latency": latency
    }

if __name__ == "__main__":
    print("Building index...")
    build_index()
    print("Done.")