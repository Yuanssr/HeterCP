LOG_DIR=opencood/logs/Baselines/stage1/OPV2V_m1_v2xvit
ERR_LOG="$LOG_DIR/train.err"
OUT_LOG="$LOG_DIR/train.log"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0 \
python -u opencood/tools/train.py -y None --model_dir "$LOG_DIR" \
  2>"$ERR_LOG" | tee "$OUT_LOG"


#Mutil gpu training
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py -y None --model_dir opencood/logs/GenComm/stage1/OPV2V_m2_att
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py -y None --model_dir opencood/logs/GenComm/stage1/OPV2V_m3_att
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py -y None --model_dir opencood/logs/GenComm/stage1/OPV2V_m4_att
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py -y None --model_dir opencood/logs/Baselines/stage1/OPV2V_m0_att

#Baseline Stage 1 Training Script

#mkdir opencood/logs/Baselines/stage1/OPV2V_m4_att

#cp opencood/hypes_yaml/opv2v/GenComm_yamls/baselines/stage1/m4_att.yaml opencood/logs/Baselines/stage1/OPV2V_m4_att/config.yaml

#CUDA_VISIBLE_DEVICES=x python opencood/tools/train.py -y None --model_dir opencood/logs/Baselines/stage1/OPV2V_m1_att/  # x is the index of GPUs

# you can also use DDP training:
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp.py -y None --model_dir opencood/logs/Baselines/stage1/OPV2V_m1_att/ 2>train.log
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp.py -y None --model_dir opencood/logs/Baselines/stage1/OPV2V_m2_att/ 2>train.log
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp.py -y None --model_dir opencood/logs/Baselines/stage1/OPV2V_m3_att/ 2>train.log
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp.py -y None --model_dir opencood/logs/Baselines/stage1/OPV2V_m4_att/ 2>train.log


