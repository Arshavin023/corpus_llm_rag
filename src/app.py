
import os
from flask import (Flask, request, jsonify, render_template,
                   Response, stream_with_context)
from dotenv import load_dotenv
from src.engine import (query_rag_stream, get_vectordb, 
                        get_llm,
                        # vectordb, 
                        build_index)

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
application = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app = application

# Optional: Ensure the index exists on first startup
if not os.path.exists("src/chroma_db"):
    print("No vectorstore found. Building index...")
    build_index()
    print("Vectorstore built successfully.")

@app.route("/")
def home():
    """Web chat interface"""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    def generate():
        # Stream RAG response including citations and safe text chunks
        for chunk in query_rag_stream(user_query):
            yield chunk  # already bytes, already JSON-safe

    return Response(
        stream_with_context(generate()),
        mimetype='application/json',
        direct_passthrough=True
    )

@app.route("/health", methods=["GET"])
def health():
    """Simple health check"""
    return jsonify({"status": "ok", "engine": "ChromaDB + Groq Llama3"})

PORT = int(os.getenv("PORT", 1762))
DEBUG = os.getenv("DEBUG", "False") == "True"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)