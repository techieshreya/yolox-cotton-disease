# YOLOX Cotton Disease Detection

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11-5C3EE8?logo=opencv&logoColor=white)
![Albumentations](https://img.shields.io/badge/Albumentations-2.0-orange)

An improved YOLOX object detector, implemented from scratch in PyTorch, for detecting and localizing cotton plant leaf diseases — including multiple co-occurring diseases at different severity stages. The work is inspired by the paper *"Handling Severity Levels of Multiple Co-Occurring Cotton Plant Diseases Using Improved YOLOX Model"* (a copy of the paper is included in this repository).

## Problem

Cotton crops are highly susceptible to leaf diseases such as leaf curl virus and sooty mold. In real field conditions, a single leaf can exhibit **multiple diseases at once and at different severity stages**, which classification-only approaches handle poorly. This project frames the task as **object detection**: each disease instance is localized with a bounding box and classified, so co-occurring diseases and their stages can be identified on the same leaf.

The model detects **5 classes**:

| Class | Description |
|---|---|
| `healthy` | Healthy leaf |
| `curl_stage1` | Leaf curl virus, early stage |
| `curl_stage2` | Leaf curl virus, advanced stage |
| `sooty` | Sooty mold |
| `leaf_enation` | Leaf enation |

## Approach

The detector is an anchor-free YOLOX built entirely from custom PyTorch modules (no dependency on the official YOLOX repo), with several architectural improvements:

- **CSPDarknet backbone** with `Focus` stem and CSP layers
- **CBAM attention** (channel + spatial attention) integrated into every CSP layer to help the network focus on subtle disease regions
- **SimSPPF** spatial pyramid pooling in the backbone, plus a standalone **Improved SPP** block (parallel 3/5/7/9 max-pooling branches) in `models/improved_ssp.py`
- **PAFPN neck** for multi-scale feature fusion and a **decoupled YOLOX head** (separate classification and regression branches) at strides 8/16/32
- **GIoU loss** for box regression combined with objectness and classification losses (`utils/loss.py`)

Training pipeline highlights:

- Heavy **Albumentations** augmentation (flips, shift-scale-rotate, color jitter, blur, Gaussian noise) with bbox-aware transforms
- **AdamW** optimizer with linear LR warmup followed by cosine annealing, and gradient clipping
- Per-epoch validation with **mAP@0.5** computed on NMS-post-processed predictions; the best checkpoint is saved automatically
- Periodic side-by-side visualizations of predictions vs. ground truth saved during training

## Dataset

The `data/` directory contains a cotton leaf disease dataset exported from **Roboflow** (1,186 source images, annotated in Pascal VOC XML, resized to 416×416, with 3× offline augmentation), split into `train` / `valid` / `test`. See `README.roboflow.txt` for the full export details. `utils/xml_to_coco.py` provides a converter from Pascal VOC XML to COCO JSON if needed; the training dataset class reads the VOC XML files directly.

## Repository structure

```
.
├── config.py               # Paths and hyperparameter defaults
├── main.py                 # CUDA availability check
├── train.py                # Training loop, validation, mAP@0.5, checkpointing
├── models/
│   ├── yolox.py            # CSPDarknet + CBAM, SimSPPF, PAFPN, decoupled head
│   └── improved_ssp.py     # Improved SPP block (3/5/7/9 max-pool pyramid)
├── utils/
│   ├── dataset.py          # Pascal VOC dataset + Albumentations pipelines
│   ├── loss.py             # YOLOX loss (GIoU + objectness + classification)
│   ├── xml_to_coco.py      # VOC XML → COCO JSON converter
│   ├── visualization.py    # Prediction visualization helpers
│   └── box_ops.py          # Box utilities
├── data/                   # Roboflow dataset (train / valid / test, VOC format)
└── *.pdf                   # Reference paper
```

## Getting started

Requires Python 3.12 and a CUDA-capable GPU (CPU works but is slow).

```bash
git clone https://github.com/techieshreya/yolox-cotton-disease.git
cd yolox-cotton-disease

# with uv (CUDA 12.8 wheels are configured in pyproject.toml)
uv sync

# or with pip
pip install torch torchvision -r requirements.txt
```

Verify your GPU is visible:

```bash
python main.py
```

> **Note:** the dataset paths in `train.py` (inside the `train()` function) are currently hardcoded to a local Windows path — point them at this repo's `data/train` and `data/valid` folders before training.

Train:

```bash
python train.py --num_classes 5 --phi s --batch_size 8 --epochs 150 --lr 1e-3
```

Key arguments: `--input_size` (default 640 640), `--phi` (model size: `s`/`m`/`l`/`x`), `--save_dir` (default `./checkpoints`), `--save_interval`. During training the script logs loss and validation mAP@0.5 each epoch, saves the best-mAP weights to `checkpoints/best_model.pth`, and periodically writes prediction visualizations.

## Tech stack

PyTorch 2.7 (CUDA 12.8) · torchvision · Albumentations · OpenCV · NumPy · SciPy · Matplotlib · tqdm · uv
