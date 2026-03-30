# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# A unified framework for LiDAR-only / Camera-only / Heterogeneous collaboration.
# Support multiple fusion strategies.


import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.box2feature_generator import Box2FeatureGenerator
from opencood.models.sub_modules.diffusion_bev_gen import DiffusionBEVGen
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, CoBEVT, Where2commFusion, Who2comFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from opencood.visualization.vis_bevfeat import vis_bev, visualize_feature_maps
from opencood.data_utils.datasets import build_dataset
import torch.nn.functional as F
import importlib
import torchvision

class HeterModelBaselineWB2FDiff(nn.Module):
    def __init__(self, args):
        super(HeterModelBaselineWB2FDiff, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        self.num_class = args['num_class'] if "num_class" in args else 1
        self.ego_modality = args['ego_modality']
        
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name
        
            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')
            a=encoder_lib.__dict__
            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls
            
            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building 
            """
            if model_setting['backbone_args'] == 'identity':
                setattr(self, f"backbone_{modality_name}", nn.Identity())
            else:
                setattr(self, f"backbone_{modality_name}", BaseBEVBackbone(model_setting['backbone_args'], 
                                                                    model_setting['backbone_args'].get('inplanes',64)))

            """
            shrink conv building
            """
            setattr(self, f"shrinker_{modality_name}", DownsampleConv(model_setting['shrink_header']))

            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
            

            setattr(self, f"cls_head_{modality_name}",nn.Conv2d(args['in_head'], args['anchor_number'] * self.num_class * self.num_class,
                                  kernel_size=1))
            setattr(self, f"reg_head_{modality_name}", nn.Conv2d(args['in_head'], 7 * args['anchor_number'] * self.num_class,
                                    kernel_size=1))
            setattr(self, f"dir_head_{modality_name}", nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                    kernel_size=1)) # BIN_NUM = 2

        """
        box to feature generator building
        """
        setattr(self, f"box2featuregenerator", Box2FeatureGenerator(model_setting['box2feature']))
        self.C_target = args['in_head']
        self.cond_channels = model_setting['box2feature']['embed_dim']
        if self.cond_channels != self.C_target:
            self.cond_align = nn.Conv2d(self.cond_channels, self.C_target, 1)
        else:
            self.cond_align = None

        self.diffusion = DiffusionBEVGen(args['diffusion_args'])

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        self.visualize_feature_flag = args["visualize_feature"]
        self.supervise_single = False
        if args.get("supervise_single", False):
            self.supervise_single = True
            in_head_single = args['in_head_single']
            setattr(self, f'cls_head_single', nn.Conv2d(in_head_single, args['anchor_number'] * self.num_class * self.num_class, kernel_size=1))
            setattr(self, f'reg_head_single', nn.Conv2d(in_head_single, args['anchor_number'] * 7 * self.num_class, kernel_size=1))
            setattr(self, f'dir_head_single', nn.Conv2d(in_head_single, args['anchor_number'] *  args['dir_args']['num_bins'], kernel_size=1))


        if args['fusion_method'] == "max":
            self.fusion_net = MaxFusion()
        if args['fusion_method'] == "att":
            self.fusion_net = AttFusion(args['att']['feat_dim'])
        if args['fusion_method'] == "disconet":
            self.fusion_net = DiscoFusion(args['disconet']['feat_dim'])
        if args['fusion_method'] == "v2vnet":
            self.fusion_net = V2VNetFusion(args['v2vnet'])
        if args['fusion_method'] == 'v2xvit':
            self.fusion_net = V2XViTFusion(args['v2xvit'])
        if args['fusion_method'] == 'cobevt':
            self.fusion_net = CoBEVT(args['cobevt'])
        if args['fusion_method'] == 'where2comm':
            self.fusion_net = Where2commFusion(args['where2comm'])
        if args['fusion_method'] == 'who2com':
            self.fusion_net = Who2comFusion(args['who2com'])


        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        
        
        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])
            self.model_train_init()
        # 冻结除 box2featuregenerator 之外的检测/特征提取模块
        self.freeze_det_modules()

        # check again which module is not fixed.
        check_trainable_module(self)

    def freeze_det_modules(self):
        """
        Freeze encoder/backbone/shrinker/cls_head/reg_head/dir_head for all modalities.
        """
        for modality_name in self.modality_name_list:
            for name in ['encoder', 'backbone', 'shrinker', 'cls_head', 'reg_head', 'dir_head']:
                m = getattr(self, f"{name}_{modality_name}", None)
                if m is not None:
                    for p in m.parameters():
                        p.requires_grad_(False)
                    m.apply(fix_bn)


    def model_train_init(self):
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)
        # 冻结除 box2featuregenerator 之外的检测/特征提取模块
        self.freeze_det_modules()


    def forward(self, data_dict):
        output_dict = {}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        # print(agent_modality_list)
        ego_ids = data_dict['ego_id_list']
        ego_flag_list = [[cid == ego_id for cid in cavs] for cavs, ego_id in zip(data_dict['cav_id_list'], ego_ids)]
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}
        # print(modality_count_dict)
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            if not isinstance(eval(f"self.backbone_{modality_name}"), nn.Identity):
                feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.shrinker_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        ego_feature = None

        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })

        """
        Assemble heter features
        """
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        # 取每个场景的 ego 特征
        ego_feats = []
        start = 0
        for rl, flags in zip(record_len, ego_flag_list):
            ego_idx = flags.index(True)          # 当前场景内 ego 的位置
            ego_feats.append(heter_feature_2d_list[start + ego_idx])
            start += rl
        ego_feature = torch.stack(ego_feats, dim=0)  # 形状: (batch, C, H, W)
        
        # 取每个场景的 non ego 特征
        gt_non_ego_feats = []
        flat_ego_flag_list = [flag for scene in ego_flag_list for flag in scene]
        for i, flags in enumerate( flat_ego_flag_list):
            if not flags:
                gt_non_ego_feats.append(heter_feature_2d_list[i])

        # 无协同车时，直接用原特征走后续流程
        if len(gt_non_ego_feats) == 0:
            heter_feature_2d = torch.stack(heter_feature_2d_list)
            heter_feature_2d_new = heter_feature_2d
            if self.compress:
                heter_feature_2d_new = self.compressor(heter_feature_2d_new)
            if self.supervise_single:
                cls_preds_before_fusion = self.cls_head_single(heter_feature_2d_new)
                reg_preds_before_fusion = self.reg_head_single(heter_feature_2d_new)
                dir_preds_before_fusion = self.dir_head_single(heter_feature_2d_new)
                output_dict.update({
                    'cls_preds_single': cls_preds_before_fusion,
                    'reg_preds_single': reg_preds_before_fusion,
                    'dir_preds_single': dir_preds_before_fusion
                })
            fused_feature = self.fusion_net(heter_feature_2d_new, record_len, affine_matrix)
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
            if self.visualize_feature_flag:
                output_dict['heter_feature_reconstruction'] = heter_feature_2d_new
                output_dict['heter_feature'] = heter_feature_2d
                output_dict['fused_feature'] = fused_feature
            ego_mod = self.ego_modality
            cls_preds = getattr(self, f"cls_head_{ego_mod}")(fused_feature)
            reg_preds = getattr(self, f"reg_head_{ego_mod}")(fused_feature)
            dir_preds = getattr(self, f"dir_head_{ego_mod}")(fused_feature)
            output_dict.update({'cls_preds': cls_preds,
                                'reg_preds': reg_preds,
                                'dir_preds': dir_preds})
            return output_dict
        
        gt_non_ego_feature = torch.stack(gt_non_ego_feats) 


        # 为每个车辆分别预测并按 cav_id 存入字典
        heter_feature_2d = torch.stack(heter_feature_2d_list)
        flat_cav_ids = [cid for scene in data_dict['cav_id_list'] for cid in scene]
        flat_modalities = agent_modality_list

        per_agent_preds = {}
        for feat, cid, modality in zip(heter_feature_2d, flat_cav_ids, flat_modalities):
            feat_b = feat.unsqueeze(0)  # 增加 batch 维度以适配卷积
            cls_head = getattr(self, f"cls_head_{modality}")
            reg_head = getattr(self, f"reg_head_{modality}")
            dir_head = getattr(self, f"dir_head_{modality}")
            per_agent_preds[cid] = {
                'cls_preds': cls_head(feat_b),
                'reg_preds': reg_head(feat_b),
                'dir_preds': dir_head(feat_b),
            }

        """
        提取车辆检测框
        """
        dataset = data_dict['dataset']
        batch_box = []
        batch_non_ego_data = data_dict['non_ego_data_list']
        for rl, cav_ids, non_ego_data, ego_id in zip(record_len, data_dict['cav_id_list'], batch_non_ego_data, ego_ids):
            sub_preds = {cid: per_agent_preds[cid] for cid in cav_ids
                         if cid  != ego_id}
            non_ego_box_list = []
            for cid in non_ego_data.keys():
                pred = {cid: sub_preds[cid]}       # 这一辆车的预测
                data = {cid: non_ego_data[cid]}    # 这一辆车的对应数据
                pred_box_tensor, pred_score, gt_box_tensor = \
                        dataset.post_process_local(data, pred)
                non_ego_box_list.append({
                    'pred_box_tensor': pred_box_tensor,
                    'pred_score': pred_score,
                    'gt_box_tensor': gt_box_tensor
                })
            batch_box.append(non_ego_box_list)


        """
        从检测框重建目标bev网格
        """
        device = ego_feature.device
        target_hw = ego_feature.shape[-2:]
        non_ego_feature_2d_list = eval(f"self.box2featuregenerator")(batch_box, device, target_hw)
        
        """
        扩散模型重建特征图
        """
        x0_list = []
        for scene_idx, flags in enumerate(ego_flag_list):  # flags: 场景内 cav 是否为 ego
            for flag in flags:
                if not flag:
                    x0_list.append(ego_feature[scene_idx])
        x0 = torch.stack(x0_list)            # [N_non_ego_agents, C, H, W]
        #gt_feature = torch.stack(heter_feature_2d_list)  # 原始各车特征，供 loss 对齐

        cond_list = []
        for non_ego_features in non_ego_feature_2d_list:
            for feat in non_ego_features:
                cond_list.append(feat)
        cond = torch.stack(cond_list)        # [N_non_ego_agents, C, H, W]
        if self.cond_align is not None:
            cond = self.cond_align(cond)

        
        gen_data_dict = self.diffusion(x0, cond)
        pred_feature = gen_data_dict['pred_feature']
        output_dict.update({'gt_feature': gt_non_ego_feature,
                            'pred_feature': pred_feature})    
        



        """
        将各个场景的非 ego 车辆特征与 ego 特征拼接还原回 heter_feature_2d_list
        """
        heter_feature_2d_list_new = []
        ptr = 0
        for scene_idx, (cav_ids, flags) in enumerate(zip(data_dict['cav_id_list'], ego_flag_list)):
            for cid, is_ego in zip(cav_ids, flags):
                if is_ego:
                    heter_feature_2d_list_new.append(ego_feature[scene_idx])
                else:
                    heter_feature_2d_list_new.append( pred_feature[ptr])
                    ptr += 1
        

        heter_feature_2d_new = torch.stack(heter_feature_2d_list_new)


        if self.compress:
            heter_feature_2d_new = self.compressor(heter_feature_2d_new)
        

        """
        Single supervision
        """
        if self.supervise_single:
            cls_preds_before_fusion = self.cls_head_single(heter_feature_2d_new)
            reg_preds_before_fusion = self.reg_head_single(heter_feature_2d_new)
            dir_preds_before_fusion = self.dir_head_single(heter_feature_2d_new)
            output_dict.update({'cls_preds_single': cls_preds_before_fusion,
                                'reg_preds_single': reg_preds_before_fusion,
                                'dir_preds_single': dir_preds_before_fusion})

        """
        Feature Fusion (multiscale).

        we omit self.backbone's first layer.
        """
        fused_feature = self.fusion_net(heter_feature_2d_new, record_len, affine_matrix)


        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)


        if self.visualize_feature_flag:
                    output_dict['heter_feature_reconstruction'] = heter_feature_2d_new
                    output_dict['heter_feature'] = heter_feature_2d
                    output_dict['fused_feature'] = fused_feature

        ego_mod = self.ego_modality  # already set in __init__
        cls_preds = getattr(self, f"cls_head_{ego_mod}")(fused_feature)
        reg_preds = getattr(self, f"reg_head_{ego_mod}")(fused_feature)
        dir_preds = getattr(self, f"dir_head_{ego_mod}")(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})


        return output_dict
