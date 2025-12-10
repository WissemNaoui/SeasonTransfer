#!/bin/bash

# Data Preparation Script for CycleGAN
# This script organizes your local data for upload to Google Drive

echo "========================================="
echo "CycleGAN Data Preparation Script"
echo "========================================="

# Source data location (your local archive)
SOURCE_DATA="/home/wissem/Downloads/archive"

# Target location (will be uploaded to Drive)
TARGET_DATA="/home/wissem/.gemini/antigravity/scratch/SeasonsGAN/data/summer2winter_yosemite"

# Create target directories
echo "Creating target directories..."
mkdir -p "$TARGET_DATA/trainA"
mkdir -p "$TARGET_DATA/trainB"
mkdir -p "$TARGET_DATA/testA"
mkdir -p "$TARGET_DATA/testB"

# Check if source exists
if [ ! -d "$SOURCE_DATA" ]; then
    echo "❌ ERROR: Source data not found at $SOURCE_DATA"
    echo "Please verify the path to your downloaded archive."
    exit 1
fi

echo "✓ Source data found"

# Copy data (adjust paths based on your archive structure)
echo "Copying data..."

# Option 1: If archive has trainA/trainB structure
if [ -d "$SOURCE_DATA/trainA" ]; then
    echo "Detected trainA/trainB structure..."
    cp -r "$SOURCE_DATA/trainA/"* "$TARGET_DATA/trainA/"
    cp -r "$SOURCE_DATA/trainB/"* "$TARGET_DATA/trainB/"
    
    if [ -d "$SOURCE_DATA/testA" ]; then
        cp -r "$SOURCE_DATA/testA/"* "$TARGET_DATA/testA/"
        cp -r "$SOURCE_DATA/testB/"* "$TARGET_DATA/testB/"
    fi
fi

# Option 2: If archive has summer/winter structure
if [ -d "$SOURCE_DATA/summer" ]; then
    echo "Detected summer/winter structure..."
    cp -r "$SOURCE_DATA/summer/"* "$TARGET_DATA/trainA/"
    cp -r "$SOURCE_DATA/winter/"* "$TARGET_DATA/trainB/"
fi

# Count images
TRAIN_A_COUNT=$(find "$TARGET_DATA/trainA" -type f | wc -l)
TRAIN_B_COUNT=$(find "$TARGET_DATA/trainB" -type f | wc -l)

echo ""
echo "========================================="
echo "Data Preparation Complete!"
echo "========================================="
echo "Summer images (trainA): $TRAIN_A_COUNT"
echo "Winter images (trainB): $TRAIN_B_COUNT"
echo ""
echo "Next steps:"
echo "1. Upload this folder to Google Drive:"
echo "   $TARGET_DATA"
echo "   → /MyDrive/SeasonsGAN/data/summer2winter_yosemite/"
echo ""
echo "2. Upload the src/ folder to Google Drive:"
echo "   $(dirname $TARGET_DATA)/../src/"
echo "   → /MyDrive/SeasonsGAN/src/"
echo ""
echo "3. Open notebooks/01_colab_train.ipynb in Google Colab"
echo "========================================="
