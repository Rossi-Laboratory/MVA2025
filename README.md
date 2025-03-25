## <b>MVA2025 Project Page</b>
[Paper]() | [Project Page](https://rossi-laboratory.github.io/MVA2025/) | [Vedio]() | [Code]()
# 🌀 DLSF Inference: Dual-Layer Synergistic Fusion for High-Fidelity Image Synthesis

This repository implements the inference pipeline for **DLSF (Dual-Layer Synergistic Fusion)**, a framework designed to enhance the image generation quality of **Stable Diffusion XL (SDXL)** models. By introducing two innovative fusion strategies — **Adaptive Global Fusion (AGF)** and **Dynamic Spatial Fusion (DSF)** — DLSF enables high-resolution, semantically-aligned, and detail-preserving image synthesis.

📄 Based on the paper:  
**"DLSF: Dual-Layer Synergistic Fusion for High-Fidelity Image Synthesis"**  
📚 Presented at: **MVA 2025**  
> This repository focuses on running inference with pretrained models. Training code is not included.

---

## 📌 Highlights

- ✅ Integrates **base latent** and **refined latent** through learnable fusion modules.
- 🎨 Supports **multi-view generation** and **prompt-based synthesis**.
- ⚙️ Built with HuggingFace `diffusers` and PyTorch, optimized for FP16 inference.
- 📈 Outperforms SDXL on FID, sFID, and Inception Score across 256×256 and 512×512 resolutions.

---

## 🧠 Technical Summary

Traditional SDXL pipelines generate a latent image from a prompt using a base model, then optionally refine it. However, feature fusion between base and refined latents is suboptimal.

DLSF solves this by:
- **AGF (Adaptive Global Fusion)**: Aligns features across semantic levels with learnable weights.
- **DSF (Dynamic Spatial Fusion)**: Applies spatial attention for pixel-level detail control.

After fusion, the latent is decoded into a **1024×1024** image using the VAE decoder.

---

## 📂 Project Structure

```
DLSF-Inference/
├── inference.ipynb         # Main Jupyter notebook with step-by-step inference
├── fusion_modules.py       # AGF and DSF fusion logic
├── requirements.txt        # Python dependency list
├── README.md               # Project documentation
└── assets/                 # Folder for output examples (optional)
```

---

## ⚙️ Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

(Optionally, install xformers for faster attention operations.)

---

## 🚀 How to Use

1. Clone the repo and open the notebook:

```bash
git clone https://github.com/your-username/DLSF-Inference.git
cd DLSF-Inference
jupyter notebook
```

2. Inside `inference.ipynb`, follow these steps:
   - Load `stable-diffusion-xl-base-1.0` and `stable-diffusion-xl-refiner-1.0`.
   - Choose a prompt (text description of the image you want).
   - Select a fusion strategy: `"AGF"` or `"DSF"`.
   - Run inference and visualize the result.

3. Example usage:

```python
prompt = "a majestic lion in a surreal cyberpunk jungle"
fusion_type = "DSF"  # or "AGF"
```

---

## 📷 Example Outputs

| Prompt | Fusion | Output |
|--------|--------|--------|
| *"a futuristic cityscape at night"* | AGF | ![example1](assets/example1_agf.jpg) |
| *"an astronaut riding a horse on Mars"* | DSF | ![example2](assets/example2_dsf.jpg) |

---

## 📊 Performance (ImageNet-Conditional)

| Method | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|--------|-------|------|--------------|-----------|
| SDXL   | 20.16 | 219.74 | 0.86 | 0.35 |
| AGF    | 18.79 | 230.43 | 0.87 | 0.39 |
| DSF    | 18.89 | 232.04 | 0.87 | 0.39 |

---

## 🧪 Environment

- Python 3.9+
- PyTorch >= 2.0
- GPU: A6000 (recommended), or any GPU with ≥24GB VRAM
- Tested with `diffusers==0.24.0`

---

## 📄 Citation

If you find this repository useful in your research, please cite the original paper:

```bibtex
@inproceedings{DLSF2025,
  title={DLSF: Dual-Layer Synergistic Fusion for High-Fidelity Image Synthesis},
  booktitle={MVA 2025},
  year={2025}
}
```

---

## 🧩 Future Work

- Extending DLSF to support video generation and 3D synthesis.
- Exploring domain-specific applications in medical imaging and remote sensing.
- Open-sourcing training pipeline for fine-tuning on custom datasets.
