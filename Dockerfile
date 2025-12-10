FROM python:3.10-slim

# Create non-root user and working directory
RUN useradd -m appuser
WORKDIR /app
USER appuser

# Install system deps required for image processing
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Copy minimal Python requirements file
COPY --chown=appuser:appuser requirements_mlops.txt ./requirements_mlops.txt

# Install CPU-only PyTorch first (keeps image small)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements_mlops.txt

# Copy only the app code we need
COPY --chown=appuser:appuser ui/ ./ui
COPY --chown=appuser:appuser api/ ./api
COPY --chown=appuser:appuser src/ ./src

# Create model folder (to be volume-mounted)
RUN mkdir -p /app/saved_models

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit UI
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
