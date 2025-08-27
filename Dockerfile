# Use NVIDIA PyTorch image (GPU-ready)
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Set working directory
WORKDIR /app

# Copy your Python scripts and requirements
COPY interface/main.py .
COPY notebooks ./Qi_notebook
COPY requirements.txt .
COPY .env .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Default command: run your main.py training script
CMD ["python", "main.py"]
