#!/bin/bash
cd /data/gestformer/src_gestformer

echo "===== [1/8] Briareo depth ====="
python main.py --hypes hyperparameters/Briareo/train_depth.json --phase train --gpu 0

echo "===== [2/8] Briareo normal ====="
python main.py --hypes hyperparameters/Briareo/train_normal.json --phase train --gpu 0

echo "===== [3/8] Briareo rgb ====="
python main.py --hypes hyperparameters/Briareo/train_rgb.json --phase train --gpu 0

echo "===== [4/8] Briareo optical flow ====="
python main.py --hypes hyperparameters/Briareo/train_optflow.json --phase train --gpu 0

echo "===== [5/8] NVGesture depth ====="
python main.py --hypes hyperparameters/NVGestures/train_depth.json --phase train --gpu 0

echo "===== [6/8] NVGesture color ====="
python main.py --hypes hyperparameters/NVGestures/train_color.json --phase train --gpu 0

echo "===== [7/8] NVGesture ir ====="
python main.py --hypes hyperparameters/NVGestures/train_ir.json --phase train --gpu 0

echo "===== [8/8] NVGesture optical flow ====="
python main.py --hypes hyperparameters/NVGestures/train_optflow.json --phase train --gpu 0

echo "모든 학습 완료!"
