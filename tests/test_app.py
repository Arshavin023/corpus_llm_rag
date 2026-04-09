import pytest
from unittest.mock import patch, MagicMock
import json
from src.app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

@patch("src.app.query_rag_stream")
def test_chat_streaming_success(mock_stream, client):
    # Mock the generator behavior of query_rag_stream
    mock_stream.return_value = [
        json.dumps({"citations": [{"source": "test.pdf"}], "content": ""}).encode("utf-8") + b"\n",
        json.dumps({"content": "The answer is 42."}).encode("utf-8") + b"\n"
    ]

    response = client.post("/chat", json={"query": "What is the meaning of life?"})
    
    assert response.status_code == 200
    # Consolidate the stream data
    data = response.data.decode("utf-8").split("\n")
    first_chunk = json.loads(data[0])
    second_chunk = json.loads(data[1])
    
    assert "citations" in first_chunk
    assert "42" in second_chunk["content"]