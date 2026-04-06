#!/bin/bash

# Create directories
mkdir -p .github/workflows
mkdir -p data
mkdir -p docs

# Create files
touch .github/workflows/main.yml
touch app.py
touch engine.py
touch eval_results.json
touch requirements.txt
touch README.md

# Optional: create placeholder files in docs
touch docs/design-and-evaluation.md
touch docs/ai-tooling.md

echo "Project structure created successfully."
