# 1. Match your local version (3.11)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 2. Install system dependencies
# Added 'libmagic-dev' which is often required by Unstructured/Magic for file parsing
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Install dependencies BEFORE copying the whole project
# This leverages Docker's cache layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application
COPY . .

# 5. Handle the database and data folders
# Ensure the appuser has permissions to write to the chroma_db
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Use the port you actually want
EXPOSE 1762

# 6. Start the application
# Increased timeout to 600 as you requested, to handle model loading
CMD ["gunicorn", "-b", "0.0.0.0:1762", "--timeout", "600", "src/app:application"]