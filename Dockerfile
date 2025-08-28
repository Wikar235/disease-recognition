# Use NVIDIA PyTorch image (GPU-ready)
FROM python:3.10

EXPOSE 8000

RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the model file
COPY best.pt .

# Copy your Python scripts
COPY interface ./interface
COPY src ./src
COPY api ./api

# Copy test images for the /test-predict endpoint
COPY test ./test

# Copy environment variables file
COPY .env .

CMD ["/bin/bash", "-c", "uvicorn api.fast:app --host 0.0.0.0 --port ${PORT:-8000}"]

# To upload a file to the API, use the following curl command:
# curl -X POST http://localhost:8000/predict -F file=@/path/to/your/image.jpg
# Or with xh (simpler):
# xh POST --form http://localhost:8000/predict file@/path/to/your/image.jpg
