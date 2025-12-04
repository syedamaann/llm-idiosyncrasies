# Which LLM Wrote This? ChatGPT, Claude, Gemini, or Grok?

https://github.com/user-attachments/assets/9de49e17-9439-43d5-9f75-ba094685ab1e

Detect which AI model (ChatGPT, Claude, Grok, Gemini, or DeepSeek) generated text using a neural network classifier.

**97.1% accuracy** | Based on [this research paper](https://arxiv.org/abs/2502.12150)

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
- CUDA GPU with 14GB+ VRAM
- ~16GB disk space (LOW_BANDWIDTH) or ~24GB (FAST_FUSED)

**Quick Start:**
```bash
# 1. Install dependencies
conda create -n llm-detector python=3.9 -y
conda activate llm-detector
pip install transformers==4.46.3 peft==0.13.2 huggingface-hub accelerate llm2vec==0.2.3

# 2. Authenticate with Hugging Face
huggingface-cli login

# 3. Download classifier (choose one)
# Option A: Minimal download (40KB)
huggingface-cli download Yida/classifier_chat head.pt --local-dir ./classifier_chat

# Option B: Pre-merged weights for faster startup (15.5GB)
huggingface-cli download Yida/classifier_chat --include "*.safetensors" --include "head.pt" --local-dir ./classifier_chat

# 4. Run classifier
python classify_text.py --checkpoint ./classifier_chat
```

**📖 For detailed instructions, troubleshooting, and examples, see [USAGE.md](USAGE.md)**

**Loading Modes:**
- **LOW_BANDWIDTH** (default): Minimal download, slower startup
- **FAST_FUSED**: Large download, faster startup

```bash
# See all options
python classify_text.py --help
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
