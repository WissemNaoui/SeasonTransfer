#!/bin/bash
# Quick start script to set up and run SeasonsGAN locally

set -e

echo "🚀 SeasonsGAN Quick Start"
echo "========================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python3 found"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate || . venv/Scripts/activate
echo "✅ Virtual environment activated"

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# Check for model checkpoints
if [ ! -d "SeasonsGAN/checkpoints" ] || [ -z "$(ls -A SeasonsGAN/checkpoints 2>/dev/null)" ]; then
    echo "⚠️  No model checkpoints found in SeasonsGAN/checkpoints/"
    echo "   Please download the trained weights from Colab first."
    echo "   Expected files: gen_summer.pth, gen_winter.pth"
fi

# Ask user what to run
echo ""
echo "What would you like to run?"
echo "1) Streamlit UI (recommended for testing)"
echo "2) FastAPI server"
echo "3) Docker (requires Docker installed)"
echo ""
read -p "Enter choice (1/2/3): " choice

case $choice in
    1)
        echo "🎨 Starting Streamlit UI..."
        streamlit run ui/app.py
        ;;
    2)
        echo "🔌 Starting FastAPI server..."
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    3)
        if ! command -v docker &> /dev/null; then
            echo "❌ Docker is not installed. Please install Docker first."
            exit 1
        fi
        echo "🐳 Starting Docker containers..."
        docker-compose up --build
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
