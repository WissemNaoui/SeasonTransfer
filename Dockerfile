FROM python:3.10-slim

# 1. Setup Root Environment
RUN useradd -m appuser
WORKDIR /app

# 2. Install System Dependencies (Required for OpenCV/Image processing)
# We include build-essential and python3-dev to compile any tricky wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Upgrade Pip & Install Torch (CPU Version)
# FIX: --extra-index-url allows pip to find flit_core on PyPI and Torch on the PyTorch repo
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Install Project Requirements
# Assuming you are using 'requirements.txt' inside your root folder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Source Code
# Copy the Python package code and the Streamlit UI
COPY --chown=appuser:appuser src/ ./src
# Copy the UI folder (our Streamlit app is at `ui/app.py`)
COPY --chown=appuser:appuser ui/ ./ui

# 6. Create Model Folder & Fix Permissions
# This ensures the folder exists and 'appuser' can read the volume mount
RUN mkdir -p /app/saved_models && chown -R appuser:appuser /app/saved_models

# 7. Switch to Non-Root User (Security Best Practice)
USER appuser

# 8. Run App
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]