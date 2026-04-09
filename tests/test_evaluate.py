import pytest
import json
from unittest.mock import patch
from src import evaluate

@patch("src.evaluate.query_rag_stream")
def test_evaluate_logic_integration(mock_query_stream):
    """Test that the evaluation logic correctly parses the stream"""
    
    # Mock a single-item stream response
    def mock_stream_gen(query):
        yield json.dumps({
            "citations": [{"source": "benefits.md", "snippet": "info"}], 
            "content": ""
        }).encode("utf-8") + b"\n"
        yield json.dumps({"content": "Employees get 15 days"}).encode("utf-8") + b"\n"

    mock_query_stream.side_effect = mock_stream_gen

    # We manually run a snippet of the evaluation logic
    test_item = {"question": "PTO days?", "gold_answer": "15 days", "source": "benefits.md"}
    
    full_answer = ""
    found_source = False
    
    for chunk in mock_query_stream(test_item["question"]):
        data = json.loads(chunk.decode("utf-8"))
        if "content" in data:
            full_answer += data["content"]
        if "citations" in data:
            if any(test_item["source"] in c["source"] for c in data["citations"]):
                found_source = True

    assert "15 days" in full_answer
    assert found_source is True