# Q-former implementation from the BLIP2 paper
 
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-WIP-yellow)]()
 
An implementation of the **Q-Former** (Querying Transformer) architecture from the [BLIP2 paper](https://arxiv.org/abs/2301.12597) for bridging frozen vision encoders and language models. Q-Former learns a fixed set of query tokens that extract relevant visual features, enabling efficient vision-language alignment with minimal trainable parameters.

This implementaion used the [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) as a frozen LLM and [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) as the frozen ViT
 
---
 
## Table of Contents
 
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Training](#training)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [References](#references)
---
 
## Overview
 
Q-Former acts as an information bottleneck between a frozen image encoder and a frozen or fine-tuned language model. A small set of learnable query tokens attend to image features via cross-attention, then feed into the language model — keeping compute costs low while retaining semantic richness.
 
**Key properties:**
- Frozen vision encoder (e.g. ViT, CLIP) — no gradient updates
- Lightweight Q-Former bridge (~188M parameters by default)
- Compatible with any autoregressive language model backbone
- Supports image-text matching, image-grounded text generation, and image-text contrastive learning
---
 
## Architecture

The Q-Former contains two transformer submodules sharing self-attention layers:
 
1. **Image Transformer** — cross-attends learned query tokens to image features from the frozen encoder
2. **Text Transformer** — processes text input; interacts with query tokens via shared self-attention
Three pre-training objectives are used jointly:
- **Image-Text Contrastive (ITC)** — aligns query output and text representations
- **Image-grounded Text Matching (ITM)** — binary classification of image-text pairs
- **Image-grounded Text Generation (ITG)** — causal language modelling conditioned on queries
---
 
## Installation
 
**Requirements:** Python 3.9+, PyTorch 2.0+, CUDA 13.0 
 
```bash
# Clone the repository
git clone https://github.com/MarwanBrt/Q-former-implementation-from-the-BLIP2-paper-
cd Q-former-implementation-from-the-BLIP2-paper-

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
 
# Install dependencies
pip install -r requirements.txt
```

---
 
## Training
 
### Pre-training
 
Pre-training uses the three joint objectives described above (ITC, ITM, ITG).

 From the Q-former-implementation-from-the-BLIP2-paper- folder
```bash
PYTHONPATH=. python3 scripts/train.py
```
 
### Dataset
 
```
RIW/small-coco
```
 
---
 
## Project Structure
 
```
Q-former-implementation-from-the-BLIP2-paper-/
├── src/multimodal_vlm
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── components.py
│   │   ├── multimodal.py
│   │   ├── qformer.py
│   │   └── trainer.py
│   ├── data
│   │   ├── __init__.py
│   │   └── datasets.py
│   └── utils.py
├── configs/
│   ├── training_config.yaml
│   ├── model_config.yaml
│   └── data_config.yaml
├── scripts/
│   └── train.py
├── train.py
├── requirements.txt
└── README.md
```

---
 
## Contributing
 
Contributions are welcome! Since this is an early-stage project, please open an issue first to discuss what you'd like to change.
 
---
 
## References
 
- Li et al. (2023). [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)
- Li et al. (2022). [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)
- [https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [https://huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
---
