from flask import Flask, request, jsonify, render_template
from src.engine import query_rag, build_index
import os
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
application = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app = application

# Ensure the index is built at least once
if not os.path.exists("chroma_db"):
    print("No database found. Building index...")
    build_index()

@app.route("/")
def home():
    """Web chat interface (Step 4)"""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """API endpoint for user questions (Step 4)"""
    data = request.json
    user_query = data.get("query", "")
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
        
    result = query_rag(user_query)
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    """Simple status check (Step 4)"""
    return jsonify({"status": "ok", "engine": "ChromaDB + Groq Llama3"})

PORT = int(os.getenv("PORT", 1762))
DEBUG = os.getenv("DEBUG", "False") == "True"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)