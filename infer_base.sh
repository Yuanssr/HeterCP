#CUDA_VISIBLE_DEVICES=6 python opencood/tools/inference.py --model_dir opencood/logs/GenComm/stage1/OPV2V_m2_att --save_vis_interval 100000
#python opencood/tools/heal_tools.py merge_and_save \
 # opencood/logs/Baselines/stage1/OPV2V_m2_att \
#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir opencood/logs/Baselines/stage2/MPDA/OPV2V_m1_V2XREAL_m5_v2xvit --visualize_feature
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir opencood/logs/Baselines/stage1/OPV2V_m5_v2xvit --visualize_feature
#CUDA_VISIBLE_DEVICES=6 python opencood/tools/inference.py --model_dir opencood/logs/Baselines/stage1/OPV2V_m1_att