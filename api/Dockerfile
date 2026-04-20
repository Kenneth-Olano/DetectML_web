FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API files and local models/DB
COPY . .

# Run the API with uvicorn (Hugging Face Spaces default port is 7860)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
