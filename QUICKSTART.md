# Quick Start Guide

Get up and running with the LLM classifier in 5 minutes.

## TL;DR

```bash
# 1. Setup environment
conda create -n llm-detector python=3.9 -y && conda activate llm-detector
pip install transformers==4.46.3 peft==0.13.2 huggingface-hub accelerate llm2vec==0.2.3

# 2. Login to Hugging Face (get token at: https://huggingface.co/settings/tokens)
huggingface-cli login

# 3. Request access to Llama 3 (visit and click "Request Access")
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

# 4. Download classifier head
huggingface-cli download Yida/classifier_chat head.pt --local-dir ./classifier_chat

# 5. Run!
python classify_text.py --checkpoint ./classifier_chat
```

---

## Common Commands

### Interactive Mode
```bash
python classify_text.py --checkpoint ./classifier_chat
```
*Prompts you to paste text, then classifies it*

### Direct Classification
```bash
python classify_text.py --checkpoint ./classifier_chat --text "Hello! I'd be happy to help you."
```
*Classifies the provided text immediately*

### Fast Mode (After Downloading Pre-Merged Weights)
```bash
# First, download merged weights (15.5GB, one-time)
huggingface-cli download Yida/classifier_chat --include "*.safetensors" --include "head.pt" --local-dir ./classifier_chat

# Then use fast mode
python classify_text.py --checkpoint ./classifier_chat --mode FAST_FUSED
```
*Starts up in 1-2 minutes instead of 5-10 minutes*

---

## Requirements Checklist

- [ ] Python 3.9 or 3.10 installed
- [ ] NVIDIA GPU with 14GB+ VRAM
- [ ] CUDA 11.8+ or 12.0+ installed
- [ ] 16GB+ system RAM
- [ ] 8GB+ free disk space
- [ ] Hugging Face account with Llama 3 access

---

## Expected Output

```
Loading classifier in LOW_BANDWIDTH mode...
🔧 Route: Building from adapters...
Loading base model...
✓ Base model loaded
✓ Moving to CPU for merging...
✓ Merging first adapter...
✓ Moving merged model back to GPU...
✓ Loading supervised adapter...
✓ Loaded classification head from ./classifier_chat/head.pt
✓ GPU 0: 13.42GB allocated, 13.98GB reserved

Enter the text to classify (press Enter twice when done):
I'd be happy to help you with that question!


Analyzing text...

==================================================
PREDICTION RESULTS
==================================================

Most likely source: Claude
Confidence: 89.23%

All probabilities:
--------------------------------------------------
Claude                89.23% ████████████████████████████████████████████
ChatGPT                6.45% ███
Gemini                 2.31% █
Grok                   1.12%
DeepSeek               0.89%
==================================================
```

---

## Troubleshooting

### "Module not found" errors
```bash
conda activate llm-detector
pip install transformers==4.46.3 peft==0.13.2 huggingface-hub accelerate llm2vec==0.2.3
```

### "Access to model is restricted"
1. Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2. Click "Request Access"
3. Run `huggingface-cli login` again

### "CUDA out of memory"
Close other programs using GPU:
```bash
nvidia-smi  # See what's using GPU
kill -9 <PID>  # Kill process if needed
```

### "head.pt not found"
```bash
huggingface-cli download Yida/classifier_chat head.pt --local-dir ./classifier_chat
```

---

## What's Next?

- **Full documentation:** See [USAGE.md](USAGE.md) for detailed instructions
- **Batch processing:** Examples in USAGE.md
- **Python API:** Integration examples in USAGE.md
- **Performance tuning:** Memory optimization tips in USAGE.md

---

## Loading Modes Compared

| Mode | Download | Startup | Use When |
|------|----------|---------|----------|
| **LOW_BANDWIDTH** | 40KB | 5-10 min | First time, testing |
| **FAST_FUSED** | 15.5GB | 1-2 min | Production, repeated use |

**Default mode:** LOW_BANDWIDTH (minimal download)

---

## Help

```bash
# Built-in help
python classify_text.py --help

# Check GPU
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check HF login
huggingface-cli whoami
```

---

## Links

- **Detailed Guide:** [USAGE.md](USAGE.md)
- **Research Paper:** https://arxiv.org/abs/2502.12150
- **Model on HuggingFace:** https://huggingface.co/Yida/classifier_chat
- **Request Llama 3 Access:** https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- **Create HF Token:** https://huggingface.co/settings/tokens

---

**Happy classifying! 🚀**
