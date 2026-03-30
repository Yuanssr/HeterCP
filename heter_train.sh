
# Take m1m2 as an example
#cp opencood/hypes_yaml/opv2v/GenComm_yamls/gencomm/stage2/m1m2_att.yaml opencood/logs/GenComm/stage2/OPV2V_m1m2_att/config.yaml

#cp opencood/hypes_yaml/opv2v/GenComm_yamls/gencomm/stage2/m1m3_att.yaml opencood/logs/GenComm/stage2/OPV2V_m1m3_att/config.yaml


#python opencood/tools/heal_tools.py merge_and_save \
  #opencood/logs/GenComm/stage1/OPV2V_m2_att \
  #opencood/logs/GenComm/stage1/OPV2V_m1_att \
  #opencood/logs/GenComm/stage2/OPV2V_m1m2_att

#python opencood/tools/heal_tools.py merge_and_save \
  #opencood/logs/GenComm/stage1/OPV2V_m3_att \
  #opencood/logs/GenComm/stage1/OPV2V_m1_att \
  #opencood/logs/GenComm/stage2/OPV2V_m1m3_att

# Train
#CUDA_VISIBLE_DEVICES=6 python opencood/tools/train.py -y None --model_dir opencood/logs/GenComm/stage2/OPV2V_m1m3_att

#Mutil gpu training
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4r --use_env opencood/tools/train_ddp.py -y None --model_dir opencood/logs/GenComm/stage2/OPV2V_m1m3_att 2>tain.log
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py -y None --model_dir opencood/logs/GenComm/stage2/OPV2V_m1m2_att