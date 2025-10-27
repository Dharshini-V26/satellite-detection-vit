# Visual Search, Retrieval & Detection in Satellite Imagery

Vision Transformer (ViT)-based system for automatic object detection and visual search in satellite imagery
This project presents a Vision Transformer-based system capable of detecting, identifying, and retrieving objects across diverse satellite images. It automatically produces labeled datasets for downstream object detection models and visual search systems.

## Project Goal

The goal of this project is to develop a comprehensive system that can:
1) Automatically detect objects in satellite imagery
2) Identify and retrieve similar objects across multiple satellite scenes
3) Generate production-ready labeled datasets
4) Produce annotations for model training and fine-tuning
5) Operate effectively across diverse satellite data sources and resolutions

## Architecture Overview

Vision Transformer for Object Detection and Retrieval
Pre-trained on ImageNet-21K
Input: 224×224 RGB satellite images
Architecture: 12 transformer blocks with 12 attention heads
Output: Bounding box coordinates and object embeddings for retrieval

INPUT SATELLITE IMAGE (224×224)
    ↓
Patch Embedding (14×14 patches)
    ↓
Vision Transformer Encoder (12 blocks)
    ↓
Multi-Head Self-Attention (global context)
    ↓
Feature Extraction + Object Localization
    ↓
Bounding Box Head (3-layer MLP)
    ↓
Embedding Head (for similarity-based retrieval)
    ↓
OUTPUT: [bbox, confidence, embedding]
        ↓
        └─→ Detection & Localization
        └─→ Visual Search & Retrieval

# Training Process
## Data Preparation
## Image Processing

Input Types: Satellite images in TIF and JPG formats (variable resolutions)
Standardization: All images resized to 224×224 pixels
Normalization: Applied ImageNet standard mean and standard deviation per channel
Format Conversion: All images standardized to JPEG for consistency

## Annotation Generation
## Methodology

Automated object detection was performed using image processing techniques to generate annotations.

## Processing pipeline includes:
Histogram Equalization: Enhances image contrast for better object visibility
Morphological Operations: Opening and closing to reduce noise and refine shapes
Edge Detection and Contour Analysis: Identifies object boundaries and spatial locations

## Output
Annotation Format: COCO-style JSON with bounding box coordinates
Dataset Quality: 3,390 detected objects across 80 satellite images
Average Objects per Image: 42.4

## Installation

git clone https://github.com/Dharshini-V26/satellite-detection-vit.git

cd satellite-detection-vit

python -m venv venv

source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

python scripts/complete_system.py

This script automatically performs:
Object detection on input satellite images
Annotation generation
Model training and fine-tuning
Production of a labeled dataset

## Individual Steps

1. Auto-Detect & Identify Objects
python scripts/generate_annotations.py

2. Train Visual Search Model
python scripts/train_vit_model.py

3. Search & Retrieve Similar Objects
python scripts/test_on_new_data.py

4. Validate Dataset Quality
python scripts/verify_predictions.py

## Key Features

1) Automatic Object Detection: Identifies objects in satellite imagery without manual labeling
2) Visual Search and Retrieval: Finds similar objects across multiple satellite scenes using learned embeddings
3) Labeled Dataset Generation: Produces automatic annotations for downstream detection models
4) Transfer Learning: Utilizes pre-trained Vision Transformer for efficient training with limited data
5) End-to-End Pipeline: Includes detection, annotation, training, and quality validation steps
6) Multi-Scale Detection: Adapts to objects of varying sizes and resolutions
