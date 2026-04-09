import pytest
import json
import os
from unittest.mock import patch, MagicMock
from src.engine import load_documents, generate_answer, query_rag_stream

# ---------------------------
# Document Loading Tests
# ---------------------------
@patch("os.listdir")
@patch("os.path.exists")
def test_load_documents_skips_unsupported(mock_exists, mock_listdir):
    """Unsupported files like .png or .exe are ignored"""
    mock_exists.return_value = True
    mock_listdir.return_value = ["policy.txt", "image.png", "data.xml"]

    with patch("src.engine.TextLoader.load") as mock_txt_load, \
         patch("src.engine.UnstructuredXMLLoader.load") as mock_xml_load:

        # Set up mock document objects
        mock_doc_txt = MagicMock()
        mock_doc_txt.page_content = "txt content"
        mock_doc_txt.metadata = {"source": "policy.txt"}
        
        mock_doc_xml = MagicMock()
        mock_doc_xml.page_content = "xml content"
        mock_doc_xml.metadata = {"source": "data.xml"}

        mock_txt_load.return_value = [mock_doc_txt]
        mock_xml_load.return_value = [mock_doc_xml]

        docs = load_documents()
        assert len(docs) == 2
        assert all(hasattr(doc, "page_content") for doc in docs)

# ---------------------------
# RAG Logic Tests (Static)
# ---------------------------
def test_generate_answer_no_context():
    """Returns guardrail message if no docs"""
    result = generate_answer("How to bake a cake?", [])
    assert result["answer"] == "I can only answer questions about company policies."
    assert result["citations"] == []

def test_generate_answer_with_context():
    """Generates answer using injected mock LLM"""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Mocked LLM Answer"
    mock_llm.invoke.return_value = mock_response

    mock_doc = MagicMock()
    mock_doc.page_content = "Employees get 10 days of leave."
    mock_doc.metadata = {"source": "manual.txt"}

    result = generate_answer("How much leave?", [mock_doc], llm_instance=mock_llm)
    assert result["answer"] == "Mocked LLM Answer"
    assert result["citations"][0]["source"] == "manual.txt"

# ---------------------------
# Streaming RAG Tests
# ---------------------------
@patch("src.engine.vectordb")
@patch("src.engine.ChatPromptTemplate.from_messages")
def test_query_rag_stream_behavior(mock_prompt, mock_vectordb):
    """Verifies that the streaming RAG yields valid JSON bytes and citations"""
    # 1. Mock Vector DB results
    mock_doc = MagicMock()
    mock_doc.page_content = "Policy details"
    mock_doc.metadata = {"source": "policy.pdf"}
    mock_vectordb.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.9)]

    # 2. Mock the Chain and its Stream
    # We mock the prompt | llm behavior
    mock_chain = MagicMock()
    mock_prompt.return_value = mock_chain
    # In LangChain, prompt | llm returns the LLM-like object, 
    # but here we just need to ensure chain.stream returns what we want
    mock_chain.__or__.return_value = mock_chain 
    
    # Create a chunk that behaves like a LangChain BaseMessageChunk
    mock_chunk = MagicMock()
    mock_chunk.content = "Streaming answer"
    mock_chain.stream.return_value = [mock_chunk]

    # 3. Execute generator
    generator = query_rag_stream("test query")
    chunks = list(generator)

    # 4. Validate output
    assert len(chunks) >= 2

    # First Chunk: Citations
    cite_data = json.loads(chunks[0].decode("utf-8"))
    assert "citations" in cite_data
    assert cite_data["citations"][0]["source"] == "policy.pdf"

    # Second Chunk: Content
    content_data = json.loads(chunks[1].decode("utf-8"))
    assert content_data["content"] == "Streaming answer"

# @patch("src.engine.vectordb")
# @patch("src.engine.llm")
# def test_query_rag_stream_behavior(mock_llm, mock_vectordb):
#     """Verifies that the streaming RAG yields valid JSON bytes and citations"""
#     # 1. Mock Vector DB results
#     mock_doc = MagicMock()
#     mock_doc.page_content = "Policy details"
#     mock_doc.metadata = {"source": "policy.pdf"}
#     # Use the method actually called in engine.py
#     mock_vectordb.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.9)]

#     # 2. Mock LLM Streaming
#     mock_chunk = MagicMock()
#     # CRITICAL: We set .content as a string so json.dumps doesn't fail on a Mock object
#     mock_chunk.content = "Streaming answer"
#     mock_llm.stream.return_value = [mock_chunk]

#     # 3. Execute generator
#     generator = query_rag_stream("test query")
#     chunks = list(generator)

#     # 4. Validate output
#     assert len(chunks) >= 2

#     # First Chunk: Citations
#     cite_data = json.loads(chunks[0].decode("utf-8"))
#     assert "citations" in cite_data
#     assert cite_data["citations"][0]["source"] == "policy.pdf"

#     # Subsequent Chunks: Content
#     content_data = json.loads(chunks[1].decode("utf-8"))
#     assert "Streaming answer" in content_data["content"]