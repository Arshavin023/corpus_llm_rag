import pytest
import os
from unittest.mock import patch, MagicMock
from src.engine import load_documents, generate_answer

# ---------------------------
# Document Loading Tests
# ---------------------------

@patch("os.listdir")
@patch("os.path.exists")
def test_load_documents_skips_unsupported(mock_exists, mock_listdir):
    """Ensure loader ignores files like .png or .exe"""
    mock_exists.return_value = True
    mock_listdir.return_value = ["policy.txt", "image.png", "data.xml"]
    
    # We mock the actual Loaders to avoid file system dependency
    with patch("src.engine.TextLoader.load") as mock_txt_load, \
         patch("src.engine.UnstructuredXMLLoader.load") as mock_xml_load:
        
        mock_txt_load.return_value = [MagicMock(page_content="txt content")]
        mock_xml_load.return_value = [MagicMock(page_content="xml content")]
        
        docs = load_documents()
        
        # Should only have 2 docs (txt and xml), ignoring the png
        assert len(docs) == 2

# ---------------------------
# RAG Logic Tests
# ---------------------------
def test_generate_answer_no_context():
    """Test guardrail when context is empty."""
    result = generate_answer("How to bake a cake?", [])
    assert result["answer"] == "I can only answer questions about company policies."
    assert result["citations"] == []

@patch("src.engine.ChatGroq")
def test_generate_answer_with_context(mock_chat_groq):
    """Test LLM chain invocation with mock context."""
    # 1. Create a specific mock for the response object
    mock_response = MagicMock()
    mock_response.content = "Mocked LLM Answer"

    # 2. Set up the LLM mock to return that response object when invoke is called
    mock_llm = mock_chat_groq.return_value
    mock_llm.invoke.return_value = mock_response

    # 3. Create mock context document
    mock_doc = MagicMock()
    mock_doc.page_content = "Employees get 10 days of leave."
    mock_doc.metadata = {"source": "manual.txt"}

    # 4. Run the function
    result = generate_answer("How much leave?", [mock_doc])

    # 5. Assertions
    assert result["answer"] == "Mocked LLM Answer"
    assert result["citations"][0]["source"] == "manual.txt"