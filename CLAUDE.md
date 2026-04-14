# CLAUDE.md — ST-MMGestFormer Project Context

This file provides full project context for Claude Code and new chat sessions.
Read this file at the start of every session before taking any action.

---

## Project Goal

Implement ST-MMGestFormer, a proposed multi-modal gesture recognition model
based on GestFormer (CVPR 2024), targeting IEEE TCSVT submission.

Proposed contributions over GestFormer baseline:
1. TSM-ResNet-18: Temporal Shift Module inserted into ResNet-18 backbone (no extra params)
2. Modality-specific Positional Encoding (MSPE): learnable PE per modality
3. Cross-Modal Attention Fusion (CMAF): feature-level cross-attention replacing late fusion

---

## Repositories

- Baseline (GestFormer reproduction): https://github.com/ygh8888/gestformer
- Proposed model (ST-MMGestFormer): https://github.com/ygh8888/Spatio-Temporal-Multi-Modal-GestFormer
- Server working directory: /data/gestformer/ (linked to ST-MMGestFormer repo)

---

## SSH Server

```
Host:     qlak315.iptime.org
Port:     20014
User:     root
```

Persistent storage: /data/ (survives container restarts)
TensorBoard: http://qlak315.iptime.org:20340 (internal 6006 → external 20340)

---

## Environment

```bash
conda activate hand_vision
```

| Item | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB MIG 1g.10gb (~9.7GB VRAM) |
| CUDA | 12.2 |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| numpy | 1.26.4 |
| opencv-python | 4.8.1.78 |

---

## Dataset Paths

### Briareo
```
/data/Briareo/
├── tof/          # depth, ir, normal modalities
│   ├── train/
│   ├── test/
│   └── validation/
├── rgb/          # rgb, optical flow modalities
│   ├── train/
│   ├── test/
│   └── validation/
└── splits/
```

### NVGesture
```
/data/NVGesture/nvgesture_arch/nvGesture_v1/
├── Video_data/
└── nvgesture_test_correct_cvpr2016_v2.lst
```

---

## Working Directory Structure

```
/data/gestformer/src_gestformer/
├── main.py
├── train.py
├── test.py                  # modified: softmax CSV export added
├── cs.py                    # modified: all combination late fusion
├── run_all_modalities.sh    # train all modalities sequentially
├── run_all_tests.sh         # test all modalities sequentially
├── models/
│   ├── temporal.py          # modified: use_tsm line removed
│   └── backbones/
│       ├── resnet.py        # current: original ResNet-18
│       └── resnet_backup.py # backup of original resnet.py
├── hyperparameters/
│   ├── Briareo/             # train_depth/normal/rgb/ir/optflow.json
│   └── NVGestures/          # train_depth/color/ir/normal/optflow.json
├── checkpoints/
│   ├── Briareo/
│   │   ├── best_train_briareo_depth.pth
│   │   ├── best_train_briareo_normal.pth
│   │   ├── best_train_briareo_rgb.pth
│   │   ├── best_train_briareo_optflow.pth
│   │   └── best_train_briareo_ir-xwavegatedffn_emb.pth
│   └── NVGestures/
│       ├── best_train_nv_depth.pth
│       ├── best_train_nv_color.pth
│       ├── best_train_nv_ir.pth
│       ├── best_train_nv_optflow.pth
│       └── best_train_nv_normal-xwavegatedffn_multi.pth
├── csv/
│   ├── Briareo/             # normal/depth/ir/rgb/rgb_optflow/original.csv
│   └── Nvgestures/          # normal/depth/ir/color/depth_optflow/original.csv
└── results/
    ├── Briareo_fusion_results.csv
    ├── Briareo_fusion_results.txt
    ├── Nvgestures_fusion_results.csv
    └── Nvgestures_fusion_results.txt
```

---

## Key Code Modifications (applied to baseline)

### models/backbones/resnet.py
- Original file backed up as resnet_backup.py
- Current resnet.py: original version (no TSM)
- resnet_tsm.py exists in outputs as reference for future TSM implementation

### models/temporal.py
- Removed: `use_tsm=True, n_frames=40` argument from backbone call (line ~23)
- Current line 22: `self.backbone = backbone(pretrained, in_planes, dropout=dropout_backbone)`

### test.py
- Added: `import pandas as pd`, `import os`
- Added: softmax probability CSV export per modality
- Added: `if i == 0:` guard for FlopCountAnalysis and summary (runs once only)
- CSV naming: `{data_type}.csv` or `{data_type}_optflow.csv`
- CSV saved to: `csv/{Dataset}/`

### cs.py
- Completely rewritten to support both Briareo and NVGesture
- Evaluates all modality combinations (1~5 modalities, 32 combinations each)
- Saves results to `results/` as both CSV and TXT
- Usage:
  ```bash
  python cs.py --dataset Briareo --all
  python cs.py --dataset Nvgestures --all
  python cs.py --dataset Briareo --modalities normal ir
  ```

---

## Baseline Results

### Single Modality — Briareo (288 test samples)

| Modality | Accuracy | Params (M) | MACs (G) |
|---|---|---|---|
| Normals | 97.57% | 24.08 | 62.74 |
| IR | 96.53% | 24.07 | 60.23 |
| RGB | 96.53% | 24.08 | 62.74 |
| Optical flow | 95.14% | 24.08 | 61.49 |
| Depth | 94.79% | 24.07 | 60.23 |

### Single Modality — NVGesture (482 test samples)

| Modality | Accuracy | Params (M) | MACs (G) |
|---|---|---|---|
| Normals | 81.95% | 24.09 | 71.57 |
| Optical flow | 81.54% | 24.08 | 35.01 |
| Depth | 78.84% | 24.08 | 68.49 |
| Color | 75.52% | 24.09 | 71.57 |
| IR | 63.28% | 24.08 | 68.49 |

### Late Fusion Best Results

| Dataset | Best Accuracy | Best Combination |
|---|---|---|
| Briareo | 98.26% | Normals + Optical flow (2 modalities) |
| NVGesture | 85.06% | All 5 modalities |

Full combination results: see results/Briareo_fusion_results.txt, results/Nvgestures_fusion_results.txt

---

## Proposed Model: ST-MMGestFormer

### Module 1 — TSM-ResNet-18
Insert Temporal Shift Module into ResNet-18 backbone.
- Target file: models/backbones/resnet.py
- Reference implementation: resnet_tsm.py (in repo outputs)
- Zero additional parameters
- Shift fraction: 1/8 of channels per layer

### Module 2 — Modality-specific Positional Encoding (MSPE)
Learnable positional encoding per modality.
- Target file: models/temporal.py
- Add modality_id argument to GestureTransformer
- nn.Embedding(num_modalities, d_model) added to transformer input

### Module 3 — Cross-Modal Attention Fusion (CMAF)
Replace late fusion (softmax averaging) with feature-level cross-attention.
- New file: models/fusion.py
- Formula:
  Q_m = F_m·W_q^m,  K_n = F_n·W_k^n,  V_n = F_n·W_v^n
  A(m→n) = softmax(Q_m·K_nᵀ/√D)·V_n
  w_m = softmax(MLP(F_m))
  F_fused = Σ_m w_m·(F_m + Σ_{n≠m} A(m→n))

### Ablation Study Plan

| ID | Model | Description |
|---|---|---|
| BL1 | GestFormer | Original baseline (single modality) |
| BL2 | GestFormer + Late Fusion | Softmax averaging (already done) |
| BL3 | + TSM | Add Temporal Shift Module |
| BL4 | + MSPE | Add modality-specific PE |
| BL5 | + CMAF | Add cross-modal attention fusion |
| BL6 | + TSM + MSPE | Combination |
| BL7 | ST-MMGestFormer | Full proposed model (TSM + MSPE + CMAF) |

---

## Next Steps

1. [ ] Implement TSM in models/backbones/resnet_tsm.py
2. [ ] Integrate TSM into temporal.py (use_tsm flag)
3. [ ] Train BL3 on Briareo and NVGesture
4. [ ] Implement MSPE in temporal.py
5. [ ] Implement CMAF in models/fusion.py
6. [ ] Add EgoGesture as third benchmark dataset
7. [ ] Run full ablation study (BL1~BL7)
8. [ ] Write IEEE TCSVT paper

---

## Important Notes

- MIG instance has ~9.7GB VRAM — batch_size=8 is the safe limit
- Always use tmux for long-running training (tmux new -s <name>)
- /data/ is persistent; everything else resets on container restart
- resnet_backup.py must not be deleted — it is the safe fallback
- temporal.py must NOT have use_tsm=True unless TSM is fully implemented
- NVGesture IR single modality is intentionally low (63.28%) — same as paper
- optical flow CSV filename: rgb_optflow (Briareo), depth_optflow (NVGesture)
