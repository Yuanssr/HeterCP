# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# A unified framework for LiDAR-only / Camera-only / Heterogeneous collaboration.
# Support multiple fusion strategies.


import torch
import torch.nn as nn
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, CoBEVT, Where2commFusion, Who2comFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn
import importlib
import torchvision


class SimpleDomainAdapter(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        hidden = int(hidden_channels) if hidden_channels is not None else int(channels)
        self.net = nn.Sequential(
            nn.Conv2d(int(channels), hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, int(channels), kernel_size=1, bias=True),
        )

    def forward(self, x):
        return x + self.net(x)


class HeterModelBaselineWDomainAdapter(nn.Module):
    def __init__(self, args):
        super(HeterModelBaselineWDomainAdapter, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()]
        self.missing_message = args.get("missing_message", False)
        self.modality_name_list = modality_name_list

        self.ego_modality = args["ego_modality"]
        self.cav_range = args["lidar_range"]
        self.sensor_type_dict = OrderedDict()
        self.visualize_feature_flag = args.get("visualize_feature", False)

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting["sensor_type"]
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting["core_method"].replace("_", "")

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            # Encoder building
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting["encoder_args"]))
            setattr(
                self,
                f"depth_supervision_{modality_name}",
                model_setting["encoder_args"].get("depth_supervision", False),
            )

            # Backbone building
            if model_setting["backbone_args"] == "identity":
                setattr(self, f"backbone_{modality_name}", nn.Identity())
            else:
                setattr(
                    self,
                    f"backbone_{modality_name}",
                    BaseBEVBackbone(
                        model_setting["backbone_args"],
                        model_setting["backbone_args"].get("inplanes", 64),
                    ),
                )

            # shrink conv building
            setattr(self, f"shrinker_{modality_name}", DownsampleConv(model_setting["shrink_header"]))

            if sensor_name == "camera":
                camera_mask_args = model_setting["camera_mask_args"]
                setattr(
                    self,
                    f"crop_ratio_W_{modality_name}",
                    (self.cav_range[3]) / (camera_mask_args["grid_conf"]["xbound"][1]),
                )
                setattr(
                    self,
                    f"crop_ratio_H_{modality_name}",
                    (self.cav_range[4]) / (camera_mask_args["grid_conf"]["ybound"][1]),
                )
                setattr(
                    self,
                    f"xdist_{modality_name}",
                    (camera_mask_args["grid_conf"]["xbound"][1] - camera_mask_args["grid_conf"]["xbound"][0]),
                )
                setattr(
                    self,
                    f"ydist_{modality_name}",
                    (camera_mask_args["grid_conf"]["ybound"][1] - camera_mask_args["grid_conf"]["ybound"][0]),
                )

        # For feature transformation
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        self.num_class = args.get("num_class", 1)
        self.supervise_single = False
        if args.get("supervise_single", False):
            self.supervise_single = True

        if args["fusion_method"] == "max":
            self.fusion_net = MaxFusion()
        if args["fusion_method"] == "att":
            self.fusion_net = AttFusion(args["att"]["feat_dim"])
        if args["fusion_method"] == "disconet":
            self.fusion_net = DiscoFusion(args["disconet"]["feat_dim"])
        if args["fusion_method"] == "v2vnet":
            self.fusion_net = V2VNetFusion(args["v2vnet"])
        if args["fusion_method"] == "v2xvit":
            self.fusion_net = V2XViTFusion(args["v2xvit"])
        if args["fusion_method"] == "cobevt":
            self.fusion_net = CoBEVT(args["cobevt"])
        if args["fusion_method"] == "where2comm":
            self.fusion_net = Where2commFusion(args["where2comm"])
        if args["fusion_method"] == "who2com":
            self.fusion_net = Who2comFusion(args["who2com"])

        # Shrink header
        self.shrink_flag = False
        if "shrink_header" in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args["shrink_header"])

        # Shared Heads
        self.cls_head = nn.Conv2d(args["in_head"], args["anchor_number"] * self.num_class * self.num_class, kernel_size=1)
        self.reg_head = nn.Conv2d(args["in_head"], 7 * args["anchor_number"] * self.num_class, kernel_size=1)
        self.dir_head = nn.Conv2d(args["in_head"], args["dir_args"]["num_bins"] * args["anchor_number"], kernel_size=1)

        # compressor will be only trainable
        self.compress = False
        if "compressor" in args:
            self.compress = True
            self.compressor = NaiveCompressor(args["compressor"]["input_dim"], args["compressor"]["compress_ratio"])

        adapter_cfg = args.get("domain_adapter", {})
        adapter_hidden = adapter_cfg.get("hidden_dim", args["in_head"])
        self.domain_adapter = SimpleDomainAdapter(args["in_head"], adapter_hidden)

        self.model_train_init_domain_adapter()
        check_trainable_module(self)

    def model_train_init_domain_adapter(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.apply(fix_bn)
        for p in self.domain_adapter.parameters():
            p.requires_grad_(True)
        self.domain_adapter.train()

    def train(self, mode=True):
        super().train(mode)
        self.apply(fix_bn)
        self.domain_adapter.train(mode)
        return self

    def _align_collaborative_features(self, heter_feature_2d, agent_modality_list, record_len):
        """
        Apply domain adapter scene-by-scene.
        The first agent in each scene is treated as ego (same convention as fusion modules).
        """
        if isinstance(record_len, torch.Tensor):
            scene_lengths = [int(x) for x in record_len.tolist()]
        else:
            scene_lengths = [int(x) for x in record_len]

        aligned_feature_list = []
        start_idx = 0
        for scene_len in scene_lengths:
            end_idx = start_idx + scene_len
            scene_feature = heter_feature_2d[start_idx:end_idx]
            scene_modalities = agent_modality_list[start_idx:end_idx]

            if scene_len <= 1:
                aligned_feature_list.append(scene_feature)
                start_idx = end_idx
                continue

            scene_ego_modality = scene_modalities[0]
            cav_local_indices = [
                local_idx
                for local_idx in range(1, scene_len)
                if scene_modalities[local_idx] != scene_ego_modality
            ]

            if len(cav_local_indices) > 0:
                cav_local_indices = torch.as_tensor(cav_local_indices, device=scene_feature.device, dtype=torch.long)
                aligned_scene_feature = scene_feature.clone()
                aligned_scene_feature[cav_local_indices] = self.domain_adapter(
                    scene_feature.index_select(0, cav_local_indices)
                )
                aligned_feature_list.append(aligned_scene_feature)
            else:
                aligned_feature_list.append(scene_feature)

            start_idx = end_idx

        if len(aligned_feature_list) == 0:
            return heter_feature_2d
        return torch.cat(aligned_feature_list, dim=0)

    def forward(self, data_dict):
        output_dict = {}
        agent_modality_list = data_dict["agent_modality_list"]
        affine_matrix = normalize_pairwise_tfm(data_dict["pairwise_t_matrix"], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict["record_len"]

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            if not isinstance(eval(f"self.backbone_{modality_name}"), nn.Identity):
                feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})["spatial_features_2d"]
            feature = eval(f"self.shrinker_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        # Crop/Padd camera feature map.
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H * eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W * eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })

        # Assemble heter features
        counting_dict = {modality_name: 0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        heter_feature_2d = self._align_collaborative_features(
            heter_feature_2d,
            agent_modality_list,
            record_len,
        )

        # Single supervision
        if self.supervise_single:
            cls_preds_before_fusion = self.cls_head(heter_feature_2d)
            reg_preds_before_fusion = self.reg_head(heter_feature_2d)
            dir_preds_before_fusion = self.dir_head(heter_feature_2d)
            output_dict.update({
                "cls_preds_single": cls_preds_before_fusion,
                "reg_preds_single": reg_preds_before_fusion,
                "dir_preds_single": dir_preds_before_fusion,
            })

        # Feature Fusion (multiscale).
        if not self.training and self.missing_message:
            missing_level = 0.05
            noise_level = 3
            print(f"Missing:{missing_level} Noise:{noise_level} inference")
            for i in range(1, heter_feature_2d.shape[0]):
                mask = torch.rand(
                    heter_feature_2d.shape[1],
                    heter_feature_2d.shape[2],
                    heter_feature_2d.shape[3],
                    device=heter_feature_2d.device,
                ) > missing_level
                noise = torch.randn_like(heter_feature_2d[i]) * noise_level
                heter_feature_2d[i] = heter_feature_2d[i] * mask + noise

        fused_feature = self.fusion_net(heter_feature_2d, record_len, affine_matrix)

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        if self.visualize_feature_flag:
            output_dict["heter_feature"] = heter_feature_2d
            output_dict["fused_feature"] = fused_feature

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({"cls_preds": cls_preds, "reg_preds": reg_preds, "dir_preds": dir_preds})

        return output_dict
