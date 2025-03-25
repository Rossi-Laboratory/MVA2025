# 🌀 DLSF Inference: Dual-Layer Synergistic Fusion for High-Fidelity Image Synthesis

> This repository provides the inference pipeline for **DLSF (Dual-Layer Synergistic Fusion)**, a framework designed to improve image generation quality in **Stable Diffusion XL (SDXL)** models using two novel fusion strategies — **Adaptive Global Fusion (AGF)** and **Dynamic Spatial Fusion (DSF)**.

---

## 🌐 Project Links

- [📄 Paper]()  |  [🔗 Project Page](https://rossi-laboratory.github.io/MVA2025/)  |  [🎞️ Video]()  |  [💻 Code]()

---

## ✨ Highlights

- 🔁 Combines **base** and **refined** latents through learnable fusion modules.
- 🧠 AGF & DSF modules preserve both global semantics and local details.
- 🎨 Supports text-to-image and multi-view generation at 1024×1024 resolution.
- ⚡ FP16-optimized inference using HuggingFace Diffusers + PyTorch.
- 📈 Improved FID, IS, and Recall over baseline SDXL on ImageNet.

---

## 🧠 Method Summary

Standard SDXL pipelines sequentially apply a base and refiner model but fuse their latents suboptimally.

**DLSF** solves this by:

- 🌀 **AGF (Adaptive Global Fusion)** — Attention-based fusion for aligning hierarchical features.
- 🧭 **DSF (Dynamic Spatial Fusion)** — Spatial attention to emphasize high-frequency details.

The resulting latent is decoded by SDXL’s VAE into a final high-resolution image.

---

## 📁 Project Structure

```
DLSF-Inference/
├── inference.ipynb         # Main inference demo (Jupyter Notebook)
├── fusion_modules.py       # AGF and DSF fusion logic
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── assets/                 # (Optional) Example output images
```

---

## ⚙️ Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Optionally, install `xformers` for faster inference:

```bash
pip install xformers
```

---

## 🚀 Running Inference

1. Clone the repository and launch Jupyter:

```bash
git clone https://github.com/your-username/DLSF-Inference.git
cd DLSF-Inference
jupyter notebook
```

2. Inside `inference.ipynb`, customize your prompt and fusion strategy:

```python
prompt = "a majestic lion in a surreal cyberpunk jungle"
fusion_type = "DSF"  # or "AGF"
```

---

## 🖼️ Output Samples

| Prompt                                 | Fusion | Output                            |
|----------------------------------------|--------|-----------------------------------|
| *a futuristic cityscape at night*      | AGF    | ![AGF](assets/example1_agf.jpg)   |
| *an astronaut riding a horse on Mars*  | DSF    | ![DSF](assets/example2_dsf.jpg)   |

---

## 📊 Performance on ImageNet (256×256 & 512×512)

| Method | FID ↓ | IS ↑     | Precision ↑ | Recall ↑ |
|--------|-------|----------|--------------|-----------|
| SDXL   | 20.16 | 219.74   | 0.86         | 0.35      |
| AGF    | 18.79 | 230.43   | 0.87         | 0.39      |
| DSF    | 18.89 | 232.04   | 0.87         | 0.39      |

---

## 💻 Environment

- Python 3.9+
- PyTorch ≥ 2.0
- GPU: NVIDIA A6000 or ≥24GB VRAM recommended
- Tested with: `diffusers==0.24.0`

---

## 📚 Citation

```bibtex
@inproceedings{DLSF2025,
  title={DLSF: Dual-Layer Synergistic Fusion for High-Fidelity Image Synthesis},
  year={2025}
}
```

---

## 🔮 Future Work

- Video and 3D content generation
- Applications in medical and industrial domains
- Releasing training pipeline for community fine-tuning
