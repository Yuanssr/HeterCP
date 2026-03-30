#CUDA_VISIBLE_DEVICES=7 python opencood/tools/inference.py --model_dir opencood/logs/late_fusion/m1m2 --fusion_method lateparallel --threshold 0.3 --suppress 0.7 

#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=4
MODEL_DIR="opencood/logs/late_fusion/m1m2"
FUSION="lateparallel"
SAVE_VIS_INTERVAL=10000

# threshold=0.3, suppress 0.5 -> 0.1
for S in $(seq 0.5 -0.1 0.1); do
  echo "==> threshold=0.3 suppress=${S}"
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python opencood/tools/inference.py \
    --model_dir "$MODEL_DIR" \
    --fusion_method "$FUSION" \
    --threshold 0.3 \
    --suppress "$S" \
    --save_vis_interval "$SAVE_VIS_INTERVAL"
done

# threshold=0.4, suppress 0.8 -> 0.1
for S in $(seq 0.8 -0.1 0.1); do
  echo "==> threshold=0.4 suppress=${S}"
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python opencood/tools/inference.py \
    --model_dir "$MODEL_DIR" \
    --fusion_method "$FUSION" \
    --threshold 0.4 \
    --suppress "$S" \
    --save_vis_interval "$SAVE_VIS_INTERVAL"
done