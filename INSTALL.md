# Installation

## Dependencies for Chat API Classification

```bash
conda create -n classification python=3.9 -y
conda activate classification
pip install llm2vec==0.2.3 tensorboard huggingface-hub
```

## Download the Classifier

```bash
bash download_classifier.sh
```

This will download the pre-trained chat API classifier (~16GB) to `models/classifier_chat`.

## System Requirements

- Python 3.9
- CUDA-capable GPU with 16GB+ VRAM (recommended)
- Alternatively, CPU (will be very slow)
- ~16GB disk space for the model