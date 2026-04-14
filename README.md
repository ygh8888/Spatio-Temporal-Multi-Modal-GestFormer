# GestFormer Baseline Reproduction

Reproduction of [GestFormer (CVPR 2024)](https://arxiv.org/abs/2312.12083) — a multi-modal gesture recognition framework based on multiscale wavelet pooling transformer. This repository serves as the baseline for subsequent research.

---

## Training Environment

| Item | Spec |
|---|---|
| **OS** | Ubuntu 20.04.3 LTS (Focal Fossa) |
| **Kernel** | Linux 5.4.0-147-generic |
| **GPU** | NVIDIA A100-SXM4-80GB (MIG 1g.10gb, ~9.7GB VRAM) |
| **CPU** | AMD EPYC 7742 64-Core Processor (128 threads) |
| **RAM** | 503 GB |
| **CUDA** | 12.2 |
| **Python** | 3.10.20 |
| **Conda env** | hand_vision |

---

## Dependencies

```bash
conda create -n hand_vision python=3.10
conda activate hand_vision
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 opencv-python==4.8.1.78 pandas imgaug einops \
    PyWavelets pytorch-wavelets tensorboardX fvcore torchinfo torchstat tqdm
```

| Package | Version |
|---|---|
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| torchaudio | 2.5.1+cu121 |
| numpy | 1.26.4 |
| opencv-python | 4.8.1.78 |
| pandas | 2.3.3 |
| imgaug | 0.4.0 |
| einops | 0.8.2 |
| PyWavelets | 1.8.0 |
| pytorch-wavelets | 1.3.0 |
| tensorboardX | 2.6.4 |
| fvcore | 0.1.5.post20221221 |
| torchinfo | 1.8.0 |
| torchstat | 0.0.7 |
| torchsummary | 1.5.1 |
| tqdm | 4.65.2 |

---

## Datasets

### Briareo
- Download from the [official page](https://aimagelab.ing.unimore.it/briareo/)
- Required directory structure:

```
/data/Briareo/
├── tof/
│   ├── train/
│   ├── test/
│   └── validation/
├── rgb/
│   ├── train/
│   ├── test/
│   └── validation/
└── splits/
```

### NVGesture
- Available on HuggingFace: `urikxx/nvgesture`
- Required directory structure:

```
/data/NVGesture/nvgesture_arch/nvGesture_v1/
├── Video_data/
└── nvgesture_test_correct_cvpr2016_v2.lst
```

---

## Training

```bash
# Generate hyperparameter JSON files
python3 generate_hypes.py

# Train all modalities sequentially
tmux new -s train
conda activate hand_vision
bash run_all_modalities.sh
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Epochs | 100 |
| LR decay steps | 50, 75 |
| Batch size | 8 |
| Backbone | ResNet-18 (ImageNet pretrained) |
| Transformer heads | 8 |
| Transformer layers | 6 |
| FFN size | 1024 |
| Dropout (2D) | 0.1 |
| Dropout (1D) | 0.5 |
| Frames per clip | 40 |

---

## Testing

```bash
# Run all modality tests and save softmax CSV files
bash run_all_tests.sh

# Results saved to: test_results.log
# Softmax CSVs saved to: csv/Briareo/, csv/Nvgestures/
```

---

## Single Modality Results

### Briareo (test set: 288 samples)

| Modality | Accuracy | Params (M) | MACs (G) |
|---|---|---|---|
| Normals | **97.57%** | 24.08 | 62.74 |
| IR | 96.53% | 24.07 | 60.23 |
| RGB | 96.53% | 24.08 | 62.74 |
| Optical flow | 95.14% | 24.08 | 61.49 |
| Depth | 94.79% | 24.07 | 60.23 |

### NVGesture (test set: 482 samples)

| Modality | Accuracy | Params (M) | MACs (G) |
|---|---|---|---|
| Normals | **81.95%** | 24.09 | 71.57 |
| Optical flow | 81.54% | 24.08 | 35.01 |
| Depth | 78.84% | 24.08 | 68.49 |
| Color | 75.52% | 24.09 | 71.57 |
| IR | 63.28% | 24.08 | 68.49 |

---

## Late Fusion Results (all modality combinations)

Late fusion is performed by averaging softmax probabilities across modalities. Run with:

```bash
python cs.py --dataset Briareo --all
python cs.py --dataset Nvgestures --all
```

### Briareo

| # | Color | Depth | IR | Normals | Opt.Flow | Accuracy |
|---|---|---|---|---|---|---|
| 1 | ✓ | | | | | 96.53% |
| 1 | | ✓ | | | | 94.79% |
| 1 | | | ✓ | | | 96.53% |
| 1 | | | | ✓ | | 97.57% |
| 1 | | | | | ✓ | 95.14% |
| 2 | ✓ | ✓ | | | | 96.18% |
| 2 | ✓ | | ✓ | | | 97.22% |
| 2 | ✓ | | | ✓ | | 97.92% |
| 2 | ✓ | | | | ✓ | 97.22% |
| 2 | | ✓ | ✓ | | | 96.88% |
| 2 | | ✓ | | ✓ | | 96.53% |
| 2 | | ✓ | | | ✓ | 96.53% |
| 2 | | | ✓ | ✓ | | 97.57% |
| 2 | | | ✓ | | ✓ | 97.92% |
| 2 | | | | ✓ | ✓ | **98.26%** |
| 3 | ✓ | ✓ | ✓ | | | 96.88% |
| 3 | ✓ | ✓ | | ✓ | | 97.22% |
| 3 | ✓ | ✓ | | | ✓ | 97.22% |
| 3 | ✓ | | ✓ | ✓ | | 97.57% |
| 3 | ✓ | | ✓ | | ✓ | 97.57% |
| 3 | ✓ | | | ✓ | ✓ | 97.92% |
| 3 | | ✓ | ✓ | ✓ | | 97.22% |
| 3 | | ✓ | ✓ | | ✓ | 97.22% |
| 3 | | ✓ | | ✓ | ✓ | 97.92% |
| 3 | | | ✓ | ✓ | ✓ | 97.92% |
| 4 | ✓ | ✓ | ✓ | ✓ | | 96.88% |
| 4 | ✓ | ✓ | ✓ | | ✓ | 97.92% |
| 4 | ✓ | ✓ | | ✓ | ✓ | 97.22% |
| 4 | ✓ | | ✓ | ✓ | ✓ | 97.92% |
| 4 | | ✓ | ✓ | ✓ | ✓ | 97.22% |
| 5 | ✓ | ✓ | ✓ | ✓ | ✓ | 97.57% |

**Best: 98.26% (Normals + Optical flow)**

### NVGesture

| # | Color | Depth | IR | Normals | Opt.Flow | Accuracy |
|---|---|---|---|---|---|---|
| 1 | ✓ | | | | | 75.52% |
| 1 | | ✓ | | | | 78.84% |
| 1 | | | ✓ | | | 63.28% |
| 1 | | | | ✓ | | 81.95% |
| 1 | | | | | ✓ | 81.54% |
| 2 | ✓ | ✓ | | | | 79.67% |
| 2 | ✓ | | ✓ | | | 77.39% |
| 2 | ✓ | | | ✓ | | 82.57% |
| 2 | ✓ | | | | ✓ | 81.33% |
| 2 | | ✓ | ✓ | | | 80.91% |
| 2 | | ✓ | | ✓ | | 82.99% |
| 2 | | ✓ | | | ✓ | 84.23% |
| 2 | | | ✓ | ✓ | | 82.78% |
| 2 | | | ✓ | | ✓ | 81.74% |
| 2 | | | | ✓ | ✓ | 84.23% |
| 3 | ✓ | ✓ | ✓ | | | 82.16% |
| 3 | ✓ | ✓ | | ✓ | | 82.57% |
| 3 | ✓ | ✓ | | | ✓ | 82.78% |
| 3 | ✓ | | ✓ | ✓ | | 83.61% |
| 3 | ✓ | | ✓ | | ✓ | 84.23% |
| 3 | ✓ | | | ✓ | ✓ | 84.65% |
| 3 | | ✓ | ✓ | ✓ | | 84.23% |
| 3 | | ✓ | ✓ | | ✓ | 84.23% |
| 3 | | ✓ | | ✓ | ✓ | 84.02% |
| 3 | | | ✓ | ✓ | ✓ | 84.85% |
| 4 | ✓ | ✓ | ✓ | ✓ | | 84.44% |
| 4 | ✓ | ✓ | ✓ | | ✓ | 84.44% |
| 4 | ✓ | ✓ | | ✓ | ✓ | 84.44% |
| 4 | ✓ | | ✓ | ✓ | ✓ | 84.65% |
| 4 | | ✓ | ✓ | ✓ | ✓ | 84.65% |
| 5 | ✓ | ✓ | ✓ | ✓ | ✓ | **85.06%** |

**Best: 85.06% (all 5 modalities)**

---

## TensorBoard

```bash
conda activate hand_vision
cd src_gestformer
tensorboard --logdir train_log --port 6006 --bind_all
```

---

## Repository Structure

```
src_gestformer/
├── main.py                        # Entry point
├── train.py                       # Training class
├── test.py                        # Test class (w/ softmax CSV export)
├── cs.py                          # Late fusion evaluation
├── run_all_modalities.sh          # Train all modalities sequentially
├── run_all_tests.sh               # Test all modalities sequentially
├── models/
│   ├── temporal.py                # GestureTransformer model
│   └── backbones/
│       ├── resnet.py              # ResNet-18 backbone
│       └── resnet_backup.py      # Original ResNet-18 (backup)
├── datasets/
│   ├── Briareo.py
│   └── NVGestures.py
├── hyperparameters/
│   ├── Briareo/                   # JSON configs per modality
│   └── NVGestures/
├── checkpoints/
│   ├── Briareo/                   # Saved model weights
│   └── NVGestures/
├── csv/
│   ├── Briareo/                   # Softmax probability CSVs
│   └── Nvgestures/
├── results/                       # Late fusion result tables
└── train_log/                     # TensorBoard logs
```

---

## Citation

```bibtex
@inproceedings{garg2024gestformer,
  title={GestFormer: Multiscale Wavelet Pooling Transformer Network for Dynamic Hand Gesture Recognition},
  author={Garg, Naveen and Gao, Mingming and Venkatesha, Yogeswara and Beerel, Peter A},
  booktitle={CVPR},
  year={2024}
}
```
