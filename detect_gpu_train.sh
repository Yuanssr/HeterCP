#!/usr/bin/env bash

set -euo pipefail

# Minimum GPUs required to start; set to 2 if you want to wait for two cards.
MIN_GPUS=1

# A GPU is considered free if util <10% and used memory <500MB.
UTIL_THRESHOLD=10
MEM_THRESHOLD=500

TRAIN_CMD="python -m torch.distributed.launch --use_env opencood/tools/train_ddp.py -y None --model_dir opencood/logs/B2F/stage2/OPV2V_m1_att"

find_free_gpus() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits |
    awk -v util="$UTIL_THRESHOLD" -v mem="$MEM_THRESHOLD" '($2 < util && $3 < mem) {print $1}'
}

while true; do
  FREE_GPUS=($(find_free_gpus))
  NUM_FREE=${#FREE_GPUS[@]}

  if [[ $NUM_FREE -ge $MIN_GPUS ]]; then
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${FREE_GPUS[*]}")
    echo "Starting with $NUM_FREE GPU(s): $CUDA_VISIBLE_DEVICES"
    python -m torch.distributed.launch --nproc_per_node="$NUM_FREE" --use_env opencood/tools/train_ddp.py -y None --model_dir opencood/logs/B2F/stage2/OPV2V_m1_att
    exit $?
  else
    echo "$(date '+%F %T') no free GPUs (need $MIN_GPUS), waiting 60s..."
    sleep 60
  fi
done