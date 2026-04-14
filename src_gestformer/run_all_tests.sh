#!/bin/bash
cd /data/gestformer/src_gestformer

LOG_FILE="test_results.log"
echo "테스트 시작: $(date)" > $LOG_FILE

run_test() {
    local label=$1
    local hypes=$2
    local resume=$3
    echo "===== $label =====" | tee -a $LOG_FILE
    python main.py --hypes $hypes --phase test --gpu 0 --resume $resume 2>&1 \
        | grep -E "^[0-9]+\.[0-9]+$|Accuracy|Inference|Total params|Total mult" \
        | head -5 | tee -a $LOG_FILE
    echo "" >> $LOG_FILE
}

run_test "[1/10] Briareo ir"         hyperparameters/Briareo/test_ir.json       checkpoints/Briareo/best_train_briareo_ir-xwavegatedffn_emb.pth
run_test "[2/10] Briareo depth"      hyperparameters/Briareo/test_depth.json    checkpoints/Briareo/best_train_briareo_depth.pth
run_test "[3/10] Briareo normal"     hyperparameters/Briareo/test_normal.json   checkpoints/Briareo/best_train_briareo_normal.pth
run_test "[4/10] Briareo rgb"        hyperparameters/Briareo/test_rgb.json      checkpoints/Briareo/best_train_briareo_rgb.pth
run_test "[5/10] Briareo optflow"    hyperparameters/Briareo/test_optflow.json  checkpoints/Briareo/best_train_briareo_optflow.pth
run_test "[6/10] NVGesture normal"   hyperparameters/NVGestures/test_normal.json checkpoints/NVGestures/best_train_nv_normal-xwavegatedffn_multi.pth
run_test "[7/10] NVGesture depth"    hyperparameters/NVGestures/test_depth.json  checkpoints/NVGestures/best_train_nv_depth.pth
run_test "[8/10] NVGesture color"    hyperparameters/NVGestures/test_color.json  checkpoints/NVGestures/best_train_nv_color.pth
run_test "[9/10] NVGesture ir"       hyperparameters/NVGestures/test_ir.json     checkpoints/NVGestures/best_train_nv_ir.pth
run_test "[10/10] NVGesture optflow" hyperparameters/NVGestures/test_optflow.json checkpoints/NVGestures/best_train_nv_optflow.pth

echo "테스트 완료: $(date)" >> $LOG_FILE
echo "모든 테스트 완료! 결과: $LOG_FILE"
