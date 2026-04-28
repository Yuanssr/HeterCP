#later fusion
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp.py -y None --model_dir  opencood/logs/late_fusion/m2
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env --master_port 29600 opencood/tools/train_ddp.py -y None --model_dir  opencood/logs/late_fusion/m3
#python opencood/tools/heal_tools.py merge_and_save \
 # opencood/logs/Baselines/stage1/OPV2V_m5_v2xvit \
  #opencood/logs/Baselines/stage1/OPV2V_m1_v2xvit \
 # opencood/logs/Baselines/stage2/Backalign/OPV2V_m1m5_v2xvit > opencood/logs/Baselines/stage2/Backalign/OPV2V_m1m5_v2xvit/merge.log

 #Offline MoE Script Generator
 python opencood/tools/heal_tools.py merge_and_save \
  opencood/logs/Baselines/stage1/OPV2V_m1_v2xvit \
   opencood/logs/Baselines/stage1/V2XREAL_m5_v2xvit \
   opencood/logs/Baselines/stage2/MPDA/V2XREAL_m5_OPV2V_m1_v2xvit > opencood/logs/Baselines/stage2/MPDA/V2XREAL_m5_OPV2V_m1_v2xvit/merge.log

