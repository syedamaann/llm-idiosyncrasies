#!/bin/bash

# Script to download the chat APIs classifier
# Classifies: ChatGPT, Claude, Grok, Gemini, DeepSeek
# Accuracy: 97.1%

echo "Downloading chat APIs classifier (ChatGPT, Claude, Grok, Gemini, DeepSeek)..."
echo "Accuracy: 97.1%"
echo ""

# Create models directory if it doesn't exist
mkdir -p models

# Download using huggingface-cli
huggingface-cli download Yida/classifier_chat --local-dir models/classifier_chat

echo ""
echo "Download complete! Classifier saved to: models/classifier_chat"
echo ""
echo "To use it, run:"
echo "python classify_text.py --checkpoint models/classifier_chat"
