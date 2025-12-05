# Which LLM Wrote This? ChatGPT, Claude, Gemini, or Grok?

https://github.com/user-attachments/assets/9de49e17-9439-43d5-9f75-ba094685ab1e

Detect which AI model (ChatGPT, Claude, Grok, Gemini, or DeepSeek) generated a piece of text using a neural network classifier.

**97.1% accuracy** | Based on [this research paper](docs/arxiv_paper.pdf)

## 🚀 Quick Start

### Option 1: Google Colab (Easiest)
1. Open `classifier.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set Runtime → Change runtime type → **T4 GPU**.
3. Run all cells.

### Option 2: Local Command Line
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Login to Hugging Face
huggingface-cli login

# 3. Run the classifier (interactive mode)
python classify_text.py --checkpoint ./classifier_chat
```

### Option 3: Local Notebooks
- **Standard**: `classifier.ipynb` (Root directory)
- **Fast/Fused**: `notebooks/classifier_fast.ipynb` (Faster startup, larger download)
- **Low Bandwidth**: `notebooks/classifier_low_bandwidth.ipynb` (Slower startup, minimal download)

---

## 🛠️ Usage Modes

The tool supports two loading modes to balance speed vs. disk usage:

| Mode | Command | Best For |
|------|---------|----------|
| **Low Bandwidth** (Default) | `python classify_text.py` | First-time users, limited disk space (<10GB) |
| **Fast Fused** | `python classify_text.py --mode FAST_FUSED` | Frequent use, production (requires ~16GB download) |

## 📂 Project Structure

- `classify_text.py`: Main CLI tool.
- `classifier.ipynb`: Interactive notebook for demos.
- `notebooks/`: Specialized notebook variants.
- `docs/`: Detailed usage guide and research papers.

## 📚 Documentation

- [Full Usage Guide](docs/USAGE.md) - Detailed setup, troubleshooting, and API docs.
- [Research Paper](docs/arxiv_paper.pdf) - Technical details on the methodology.

## 📜 Citation

```bibtex
@article{sun2025idiosyncrasies,
    title    = {Idiosyncrasies in Large Language Models},
    author   = {Sun, Mingjie and Yin, Yida and Xu, Zhiqiu and Kolter, J. Zico and Liu, Zhuang},
    year     = {2025},
    journal  = {arXiv preprint arXiv:2502.12150}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
