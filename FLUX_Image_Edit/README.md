<div align="center">

# FlowSteer: Conditioning Flow Field for Consistent Image Restoration

[![Paper](https://img.shields.io/badge/arXiv-2512.08125-b31b1b.svg)](https://arxiv.org/abs/2512.08125)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://tharindu-nirmal.github.io/FlowSteer/)
[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-green.svg)](https://cvpr.thecvf.com/)

**[Tharindu Wickremasinghe](https://github.com/Tharindu-Nirmal) · Chenyang Qi · Harshana Weligampola · Zhengzhong Tu · Stanley H. Chan**

Purdue University &nbsp;·&nbsp; HKUST &nbsp;·&nbsp; Texas A&M University

</div>

---

FlowSteer is an operator-aware conditioning method that enables flow-based generative models (FLUX) to perform **zero-shot image restoration** — super-resolution, deblurring, denoising, and colorization — without retraining or task-specific adapters. The method injects a measurement prior along the sampling trajectory at each step, steering the flow toward clean images that are consistent with the degraded observation.

For visual results and comparisons, see the **[project page](https://tharindu-nirmal.github.io/FlowSteer/)**.

---

## Setup

### 1. Create the environment

```bash
conda create -n flowsteer python=3.10
conda activate flowsteer
pip install -r requirements-full.txt
pip install -e .        # installs the local flux package
```

### 2. Download the FLUX.1-dev model

Model weights are downloaded automatically from Hugging Face on the first run. Before running:

1. Accept the [FLUX.1-dev license](https://huggingface.co/black-forest-labs/FLUX.1-dev) on Hugging Face.
2. Log in to the Hub:

```bash
huggingface-cli login
```

If you already have the weights stored locally, point to them with environment variables instead:

```bash
export FLUX_DEV=/path/to/flux1-dev.safetensors
export AE=/path/to/ae.safetensors
```

---

## Quick Demo

Two sample images and their pre-generated degraded versions are included in `demo/`. Run all four restoration tasks on one of them directly:

```bash
cd src
bash batch_run.sh
```

Results are saved to `demo/outputs/`.

---

## Restore Your Own Image

**Step 1** — generate degraded versions of your clean image.

Place your image(s) in `demo/inputs/` and run:

```bash
python demo/prepare_demo.py
```

This writes four degraded versions per image to `demo/degraded/` (super-resolution, colorization, denoising, deblurring).

**Step 2** — run FlowSteer on a degraded image:

```bash
cd src
python edit_image.py \
    --source_prompt "A low resolution image of a cat." \
    --target_prompt "A high resolution image of a cat. Sharp fur details, photorealistic." \
    --degradation "super resolution" \
    --guidance 4 \
    --inject 5 \
    --source_img_dir ../demo/degraded/superres_4x/your_image.jpg \
    --output_dir ../demo/outputs/ \
    --num_steps 30 --name flux-dev --offload
```

### Supported Tasks

| Task | `--degradation` value |
|------|-----------------------|
| Super-resolution (4×) | `super resolution` |
| Deblurring | `deblurring` |
| Denoising | `denoising` |
| Colorization | `colorization` |

### Key Parameters

| Parameter | Typical range | Description |
|-----------|--------------|-------------|
| `--inject` | 4 – 9 | FlowSteer conditioning steps — higher values enforce the measurement prior more strongly |
| `--guidance` | 3 – 5 | Classifier-free guidance scale |
| `--num_steps` | 25 – 30 | Total flow steps |
| `--lambda_start` / `--lambda_step` / `--lambda_end` | 0.0 – 1.0 | Fraction of denoising steps over which the DDNM correction is active. **These should be tuned per degradation task** — the values in `batch_run.sh` provide task-specific starting points for super-resolution, colorization, denoising, and deblurring |

---

## Citation

If you find this work helpful, please star this repo and cite our paper.

```bibtex
@article{Tharindu2025_FlowSteer,
  title   = {FlowSteer: Conditioning Flow Field for Consistent Image Restoration},
  author  = {Wickremasinghe, Tharindu and Qi, Chenyang and Weligampola, Harshana and Tu, Zhengzhong and Chan, Stanley H.},
  journal = {arXiv preprint arXiv:2512.08125},
  year    = {2025},
}
```
