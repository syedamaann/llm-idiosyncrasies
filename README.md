# LLM Detector

Detect which AI model (ChatGPT, Claude, Grok, Gemini, or DeepSeek) generated text using a neural network classifier.

**97.1% accuracy** on test data | Based on [this research paper](https://arxiv.org/abs/2502.12150)

## Quick Start

### Option 1: Google Colab

1. Upload `classifier.ipynb` to [Google Colab](https://colab.research.google.com)
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Add HuggingFace token to Colab Secrets (see notebook for details)
4. Run all cells

### Option 2: Kaggle Notebooks

1. Upload `classifier.ipynb` to [Kaggle Notebooks](https://kaggle.com)
2. Enable GPU: Settings → Accelerator → GPU P100
3. Add HuggingFace token to Kaggle Secrets (see notebook for details)
4. Run all cells

### Option 3: Local Setup

**Requirements:**
- Python 3.9+
- CUDA GPU with 16GB+ VRAM
- ~16GB disk space

**Install:**
```bash
conda create -n llm-detector python=3.9 -y
conda activate llm-detector
pip install llm2vec==0.2.3 tensorboard huggingface-hub
```

**Download classifier:**
```bash
bash download_classifier.sh
```

**Run:**
```bash
python classify_text.py --checkpoint models/classifier_chat
```

Or provide text directly:
```bash
python classify_text.py --checkpoint models/classifier_chat --text "Your text here"
```

## How It Works

The classifier uses [LLM2Vec](https://arxiv.org/abs/2404.05961) (fine-tuned Llama 3 8B) to:
1. Convert text into embeddings capturing unique writing patterns
2. Classify which of 5 models generated the text
3. Output confidence scores

## Limitations

- Works best on unedited AI-generated text
- Accuracy decreases for:
  - Very short text (< 50 words)
  - Heavily edited content
  - Translated text

## Citation

```bibtex
@article{sun2025idiosyncrasies,
    title    = {Idiosyncrasies in Large Language Models},
    author   = {Sun, Mingjie and Yin, Yida and Xu, Zhiqiu and Kolter, J. Zico and Liu, Zhuang},
    year     = {2025},
    journal  = {arXiv preprint arXiv:2502.12150}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details
