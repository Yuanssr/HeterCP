LOG_DIR=opencood/logs/Baselines/stage1/V2XREAL_m1_v2xvit
ERR_LOG="$LOG_DIR/train.err"
OUT_LOG="$LOG_DIR/train.log"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 \
torchrun --nproc_per_node=4 \
  opencood/tools/train_ddp.py -y None --model_dir "$LOG_DIR" \
  2>"$ERR_LOG" | tee "$OUT_LOG"

#CUDA_VISIBLE_DEVICES=4  \
#python -u opencood/tools/train.py -y None --model_dir "$LOG_DIR" \
  #2>"$ERR_LOG" | tee "$OUT_LOG"

## BackAlign & STAMP style
## CodeFilling & MPDA style
#mkdir opencood/logs/Baselines/stage2
#mkdir opencood/logs/Baselines/stage2/MPDA
#mkdir opencood/logs/Baselines/stage2/MPDA/OPV2V_m1m2_att
#mkdir -p opencood/logs/Baselines/stage2/BackAlign/OPV2V_m1m2_att
#mkdir -p opencood/logs/Baselines/stage2/CodeFilling/OPV2V_m1m2_att
#mkdir -p opencood/logs/Baselines/stage2/STAMP/OPV2V_m0m1_att
# Take m1m2m3 as an example
#scp -r -P 25485 /home/dancer/GenComm root@202.114.0.141:/root
# copy config.yamlGenComm/train.log
#cp opencood/hypes_yaml/opv2v/GenComm_yamls/baselines/stage2/STAMP/m0m1_att.yaml opencood/logs/Baselines/stage2/STAMP/OPV2V_m0m1_att

#python opencood/tools/heal_tools.py merge_and_save \
 # opencood/logs/Baselines/stage1/OPV2V_m1_att \
 #opencood/logs/Baselines/stage1/OPV2V_m0_att \
 # opencood/logs/Baselines/stage2/STAMP/OPV2V_m0m1_att


# `python opencood/tools/heal_tools.py merge_and_save` will automatically search the best checkpoints for each folder and merge them together. The collaboration base's folder (m1 here) should be put in the second to last place, while the output folder should be put last.

# Then you can train new agent type as below:
#python opencood/tools/train.py -y None --model_dir opencood/logs/Baselines/stage2/MPDA/OPV2V_m1m3_att # you can also use DDP training

#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_stamp.py -y None --model_dir opencood/logs/Baselines/stage2/STAMP/OPV2V_m0m1_att 2>&1 train.log 
#CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py -y None --model_dir opencood/logs/Baselines/stage2/FullFt/OPV2V_m1m2_att

## BackAlign & STAMP style
#mkdir opencood/logs/Baselines/stage2/BackAlign/OPV2V_m1m2_att
#mkdir opencood/logs/Baselines/stage2/BackAlign/OPV2V_m1m3_att
#mkdir opencood/logs/Baselines/stage2/BackAlign/OPV2V_m1m4_att

# Take m1m3 as an example 
# copy config.yaml
#cp opencood/hypes_yaml/opv2v/GenComm_yamls/baselines/stage2/BackAlign/m1m3_att.yaml opencood/logs/Baselines/stage2/BackAlign/OPV2V_m1m3_att/config.yaml

# combine ckpt form stage1
#python opencood/tools/heal_tools.py merge_and_save \
 # opencood/logs/Baselines/stage1/OPV2V_m3_att \
  #opencood/logs/Baselines/stage1/OPV2V_m1_att \
 # opencood/logs/Baselines/stage2/BackAlign/OPV2V_m1m3_att

# Make sure that ego_dir is placed as the second-to-last argument, and the directory for saving the combined checkpoint is placed as the last argument.

# Then you can train new agent type as below:

#python opencood/tools/train.py -y None --model_dir opencood/logs/Baselines/stage2/BackAlign/OPV2V_m1m3_att # you can also use DDP training


