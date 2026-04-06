import pytest
from unittest.mock import patch
from src.app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

# ---------------------------
# Basic Route Tests
# ---------------------------

def test_home_page(client):
    """Test that the index page loads."""
    response = client.get("/")
    assert response.status_code == 200

def test_health_route(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"
    assert "engine" in response.json

# ---------------------------
# Chat API Tests
# ---------------------------

@patch("src.app.query_rag")
def test_chat_post_success(mock_query, client):
    """Test successful chat interaction."""
    # Mock return value from engine.py
    mock_query.return_value = {
        "query": "What is PTO?",
        "answer": "15 days.",
        "citations": [{"source": "benefits.md", "snippet": "PTO info"}],
        "latency": 0.5
    }

    response = client.post(
        "/chat",
        json={"query": "What is PTO?"}
    )

    assert response.status_code == 200
    assert response.json["answer"] == "15 days."
    assert response.json["citations"][0]["source"] == "benefits.md"

def test_chat_no_query(client):
    """Test error handling when no query is provided."""
    response = client.post("/chat", json={})
    assert response.status_code == 400
    assert "error" in response.json