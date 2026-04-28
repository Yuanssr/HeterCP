LOG_DIR=opencood/logs/GenComm/stage2/OPV2V_m1m5_v2xvit
ERR_LOG="$LOG_DIR/train.err"
OUT_LOG="$LOG_DIR/train.log"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0,5,6,7 PYTHONUNBUFFERED=1 \
torchrun --nproc_per_node=4 \
  opencood/tools/train_ddp.py -y None --model_dir "$LOG_DIR" \
  2>"$ERR_LOG" | tee "$OUT_LOG"

#推理GenComm/OPV2V_m1m5_v2xvit
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir opencood/logs/GenComm/stage2/OPV2V_m1m5_v2xvit --visualize_feature