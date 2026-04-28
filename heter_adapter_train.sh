#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=opencood/logs/Baselines/stage2/Adapter/OPV2V_m1_V2XREAL_m5_v2xvit
ERR_LOG="$LOG_DIR/train.err"
OUT_LOG="$LOG_DIR/train.log"
mkdir -p "$LOG_DIR"

# smoke run with single process; config is read from $LOG_DIR/config.yaml via -y None --model_dir
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/dancer/miniconda3/bin/conda run -n heal --no-capture-output \
  torchrun --nproc_per_node=1 \
  opencood/tools/train_ddp.py -y None --model_dir "$LOG_DIR" --half \
  2>"$ERR_LOG" | tee "$OUT_LOG"
