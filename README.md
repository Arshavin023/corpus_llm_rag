# corpus_llm_rag

mkdir -p ~/.streamlit
nano ~/.streamlit/config.toml

[server]
headless = true
# This stops it from scanning your whole venv
runOnSave = true
[browser]
gatherUsageStats = false
[logger]
level = "error"

streamlit run app.py --server.fileWatcherType none