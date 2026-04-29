#CUDA_VISIBLE_DEVICES=6 python opencood/tools/inference.py --model_dir opencood/logs/GenComm/stage1/OPV2V_m2_att --save_vis_interval 100000
#python opencood/tools/heal_tools.py merge_and_save \
 # opencood/logs/Baselines/stage1/OPV2V_m2_att \
#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir opencood/logs/Baselines/stage2/MPDA/OPV2V_m1_V2XREAL_m5_v2xvit --visualize_feature
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_v2xreal.py --model_dir opencood/logs/Baselines/stage2/MPDA/V2XREAL_m5_OPV2V_m1_v2xvit --visualize_feature
#CUDA_VISIBLE_DEVICES=6 python opencood/tools/inference.py --model_dir opencood/logs/Baselines/stage1/OPV2V_m1_att

rsync -avP -e "ssh -p 50005 -i /home/dancer/.ssh/sg-docker-public-rsa" /home/dancer/HeterCP/opencood/dataset/v2xreal dancer@222.20.98.174:/home/dancer/HEAL/dataset

#scp -P 50005 -i /home/dancer/.ssh/sg-docker-public-rsa /home/dancer/HeterCP/opencood/logs/Baselines/stage2/Backalign/V2XREAL_m5_OPV2V_m1_v2xvit/net_epoch1.pth dancer@222.20.98.174:/home/dancer/HeterCP/opencood/logs/Baselines/stage2/Backalign/V2XREAL_m5_OPV2V_m1_v2xvit/ 