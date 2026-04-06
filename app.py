from flask import Flask, request, jsonify
import streamlit as st
from engine import query_rag

# -------------------
# FLASK API
# -------------------
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")

    result = query_rag(query)
    return jsonify(result)


# -------------------
# STREAMLIT UI
# -------------------
def run_streamlit():
    st.title("📚 Policy RAG Assistant")

    query = st.text_input("Ask about company policies:")

    if query:
        result = query_rag(query)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Citations")
        for c in result["citations"]:
            st.write(f"**Source:** {c['source']}")
            st.write(c["snippet"])
            st.write("---")

        st.write(f"⏱ Latency: {result['latency']:.2f}s")


if __name__ == "__main__":
    run_streamlit()