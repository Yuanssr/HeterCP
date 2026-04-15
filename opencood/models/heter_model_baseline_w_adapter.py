# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, CoBEVT, Where2commFusion, Who2comFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn
import importlib
import torchvision

class LightweightAdapter(nn.Module):
    """
    轻量级特征域适配器，具有极少的参数量。
    在保证特征维度不变的前提下，通过1x1卷积进行跨域的语义对齐。
    """
    def __init__(self, in_channels):
        super(LightweightAdapter, self).__init__()
        # 使用 1x1 卷积和残差结构，参数量极低
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1)
        )

    def forward(self, x):
        # 残差连接，帮助保留原始语义信息
        return x + self.adapter(x)


class HeterModelBaselineWAdapter(nn.Module):
    def __init__(self, args):
        super(HeterModelBaselineWAdapter, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.missing_message = args.get('missing_message', False)
        self.modality_name_list = modality_name_list
        
        # In this adapter approach, we will freeze ALMOST EVERYTHING
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

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            # Encoder building
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            setattr(self, f"depth_supervision_{modality_name}", model_setting['encoder_args'].get("depth_supervision", False))

            # Backbone building 
            if model_setting['backbone_args'] == 'identity':
                setattr(self, f"backbone_{modality_name}", nn.Identity())
            else:
                setattr(self, f"backbone_{modality_name}", BaseBEVBackbone(model_setting['backbone_args'], 
                                                                       model_setting['backbone_args'].get('inplanes',64)))
            
            # shrink conv building
            setattr(self, f"shrinker_{modality_name}", DownsampleConv(model_setting['shrink_header']))

            # === [NEW] Lightweight Adapter building ===
            # Assigning an adapter for each modality
            in_channels = args['in_head'] # Assume shrinker outputs the final dimension before fusion
            setattr(self, f"adapter_{modality_name}", LightweightAdapter(in_channels))

            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))

        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1
        self.num_class = args.get('num_class', 1)
        self.visualize_feature_flag = args.get("visualize_feature", False)

        # Fusion network building
        fusion_map = {
            "max": MaxFusion, "att": lambda: AttFusion(args['att']['feat_dim']),
            "disconet": lambda: DiscoFusion(args['disconet']['feat_dim']),
            "v2vnet": lambda: V2VNetFusion(args['v2vnet']), "v2xvit": lambda: V2XViTFusion(args['v2xvit']),
            "cobevt": lambda: CoBEVT(args['cobevt']), "where2comm": lambda: Where2commFusion(args['where2comm']),
            "who2com": lambda: Who2comFusion(args['who2com'])
        }
        self.fusion_net = fusion_map[args['fusion_method']]() if args['fusion_method'] != 'max' else MaxFusion()

        # Shared Heads
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'] * self.num_class * self.num_class, kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'] * self.num_class, kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1)

        # Initialize parameter freezing
        self.model_train_init_adapter()
        check_trainable_module(self)

    def model_train_init_adapter(self):
        """
        冻结除了 adapter 以外的所有模块。
        """
        # 1. 冻结所有参数并固定 BN
        for p in self.parameters():
            p.requires_grad_(False)
        self.apply(fix_bn)

        # 2. 仅解冻 adapter 及其 BN
        for modality_name in self.modality_name_list:
            adapter = getattr(self, f"adapter_{modality_name}", None)
            if adapter is not None:
                # 恢复 param
                for p in adapter.parameters():
                    p.requires_grad_(True)
                
                # 定义解除 BN fix 的函数
                def unfix_bn(m):
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.train() # 让 BN 层可以在训练时更新统计量
                adapter.apply(unfix_bn)

    def forward(self, data_dict):
        output_dict = {}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        # 1. 特征编码
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            
            # 因为 encoder, backbone 和 shrinker 在训练时是固定权重的（甚至可以加 with torch.no_grad() 减少显存消耗）
            # 但是由于我们是通过 requires_grad=False 处理的，pytorch自动不计算不需要梯度的节点
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            if not isinstance(eval(f"self.backbone_{modality_name}"), nn.Identity):
                feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.shrinker_{modality_name}")(feature)
            
            # === [NEW] 通过轻量级适配器映射到统一的语义域 ===
            feature = eval(f"self.adapter_{modality_name}")(feature)
            
            modality_feature_dict[modality_name] = feature

        # 2. Camera Padding
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)

        # 3. 组合异构特征
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        # 4. 特征融合
        fused_feature = self.fusion_net(heter_feature_2d, record_len, affine_matrix)

        if self.visualize_feature_flag:
            output_dict['heter_feature'] = heter_feature_2d
            output_dict['fused_feature'] = fused_feature

        # 5. 检测头预测
        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})

        return output_dict