# Chat API LLM Detector

Detect which chat API (ChatGPT, Claude, Grok, Gemini, or DeepSeek) generated a piece of text using a pre-trained neural network classifier.

Based on the research paper: [**Idiosyncrasies in Large Language Models**](https://arxiv.org/abs/2502.12150)

## Accuracy

**97.1%** accuracy on test data (1,000 samples per model)

## Quick Start

### 1. Install Dependencies

```bash
conda create -n classification python=3.9 -y
conda activate classification
pip install llm2vec==0.2.3 tensorboard huggingface-hub
```

### 2. Download the Pre-trained Classifier

```bash
bash download_classifier.sh
```

This downloads the classifier (~16GB) to `models/classifier_chat`.

### 3. Classify Text

**Interactive mode** (paste text when prompted):
```bash
python classify_text.py --checkpoint models/classifier_chat
```

**Direct input**:
```bash
python classify_text.py --checkpoint models/classifier_chat --text "Your text here"
```

## Example Output

```
==================================================
PREDICTION RESULTS
==================================================

Most likely source: Claude
Confidence: 89.34%

All probabilities:
--------------------------------------------------
Claude               89.34% ████████████████████████████████████████████
ChatGPT               7.23% ███
Gemini                2.15% █
DeepSeek              0.89%
Grok                  0.39%
==================================================
```

## How It Works

The classifier uses [LLM2Vec](https://arxiv.org/abs/2404.05961) (a fine-tuned Llama 3 model) to:
1. Convert text into embeddings that capture unique writing style patterns
2. Classify which of the 5 chat APIs generated the text
3. Output confidence scores for each model

## Limitations

- Works best on unedited LLM-generated text
- Accuracy may decrease for:
  - Very short text (< 50 words)
  - Heavily edited or paraphrased text
  - Translated text
  - Plain factual content without stylistic markers

## Requirements

- Python 3.9
- CUDA-capable GPU with 16GB+ VRAM (recommended)
- Or CPU (very slow)

## Citation

If you use this classifier, please cite:

```bibtex
@article{sun2025idiosyncrasies,
    title    = {Idiosyncrasies in Large Language Models},
    author   = {Sun, Mingjie and Yin, Yida and Xu, Zhiqiu and Kolter, J. Zico and Liu, Zhuang},
    year     = {2025},
    journal  = {arXiv preprint arXiv:2502.12150}
}
```

## License

This project is released under the MIT license. See [LICENSE](LICENSE) for details.

## Original Research

- **Paper**: [Idiosyncrasies in Large Language Models](https://arxiv.org/abs/2502.12150)
- **Project Page**: [https://eric-mingjie.github.io/llm-idiosyncrasies/](https://eric-mingjie.github.io/llm-idiosyncrasies/)
- **Pre-trained Model**: [Yida/classifier_chat](https://huggingface.co/Yida/classifier_chat)

## Contact

For questions about the original research:
- mingjies at cs.cmu.edu
- davidyinyida0609 at berkeley.edu
