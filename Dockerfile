# Stage 1: Build the frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy frontend source and install dependencies
COPY frontend/package*.json ./
RUN npm ci

# Copy the rest of the frontend code and build
COPY frontend/ ./
RUN npm run build

# Stage 2: Serve backend + frontend
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API files and local models/DB
COPY api/ .

# Copy built frontend from Stage 1 into the `static` directory
COPY --from=frontend-builder /app/frontend/dist ./static

# Run the API with uvicorn (Hugging Face Spaces default port is 7860)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
