# -*- coding: utf-8 -*-

import importlib
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision

from opencood.models.fuse_modules.fusion_in_one import (
    AttFusion,
    CoBEVT,
    DiscoFusion,
    MaxFusion,
    V2VNetFusion,
    V2XViTFusion,
    Where2commFusion,
    Who2comFusion,
)
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.domain_adapter import HeterDomainAdapter
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.utils.model_utils import check_trainable_module, fix_bn
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class HeterModelBaselineWAdapter(nn.Module):
    @staticmethod
    def _infer_shrunk_channels(model_setting: Dict, fallback: int) -> int:
        shrink_header = model_setting.get("shrink_header", None)
        if isinstance(shrink_header, dict):
            dim_cfg = shrink_header.get("dim", None)
            if isinstance(dim_cfg, list) and len(dim_cfg) > 0:
                return int(dim_cfg[-1])
        return int(fallback)

    @staticmethod
    def _unfix_bn(module: nn.Module) -> None:
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.train()

    @staticmethod
    def _resolve_source_modalities(
        modality_name_list: List[str],
        target_modality: str,
        source_cfg: Optional[object],
    ) -> List[str]:
        if source_cfg is None:
            source_modalities = [m for m in modality_name_list if m != target_modality]
        elif isinstance(source_cfg, str):
            source_modalities = [source_cfg]
        else:
            source_modalities = list(source_cfg)

        filtered: List[str] = []
        for modality_name in source_modalities:
            if modality_name in modality_name_list and modality_name != target_modality and modality_name not in filtered:
                filtered.append(modality_name)
        return filtered

    def __init__(self, args):
        super(HeterModelBaselineWAdapter, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()]
        self.missing_message = args.get("missing_message", False)
        self.modality_name_list = modality_name_list

        self.ego_modality = args["ego_modality"]
        self.cav_range = args["lidar_range"]
        self.sensor_type_dict = OrderedDict()

        self.domain_adapter_cfg = args.get("domain_adapter", {})
        self.target_modality = args.get("target_modality", self.ego_modality)
        source_cfg = self.domain_adapter_cfg.get("source_modalities", None)
        if source_cfg is None:
            source_cfg = self.domain_adapter_cfg.get("source_modality", args.get("source_modality", None))
        self.source_modalities = self._resolve_source_modalities(
            self.modality_name_list,
            self.target_modality,
            source_cfg,
        )
        if len(self.source_modalities) == 0:
            raise ValueError("No valid source modalities for domain adaptation.")

        self.detach_target_feature = bool(self.domain_adapter_cfg.get("detach_target_feature", True))

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

        source_channel_cfg = self.domain_adapter_cfg.get(
            "c_source",
            self.domain_adapter_cfg.get("c5", None),
        )
        target_channels = int(
            self.domain_adapter_cfg.get(
                "c_target",
                self.domain_adapter_cfg.get("c1", args.get("in_head", 64)),
            )
        )
        if target_channels != int(args.get("in_head", target_channels)):
            raise ValueError("domain_adapter.c1 must match in_head for fusion compatibility.")

        target_size_cfg: Optional[Tuple[int, int]] = None
        if isinstance(self.domain_adapter_cfg.get("target_size", None), (list, tuple)):
            size_cfg = self.domain_adapter_cfg["target_size"]
            if len(size_cfg) == 2:
                target_size_cfg = (int(size_cfg[0]), int(size_cfg[1]))

        self.domain_adapters = nn.ModuleDict()
        for source_modality in self.source_modalities:
            source_model_cfg = args.get(source_modality, {})
            if isinstance(source_channel_cfg, dict):
                source_channels = int(
                    source_channel_cfg.get(
                        source_modality,
                        self._infer_shrunk_channels(source_model_cfg, args.get("in_head", 64)),
                    )
                )
            elif source_channel_cfg is not None:
                source_channels = int(source_channel_cfg)
            else:
                source_channels = int(self._infer_shrunk_channels(source_model_cfg, args.get("in_head", 64)))

            self.domain_adapters[source_modality] = HeterDomainAdapter(
                c_source=source_channels,
                c_target=target_channels,
                target_size=target_size_cfg,
                alpha=float(self.domain_adapter_cfg.get("alpha", 0.1)),
                eps=float(self.domain_adapter_cfg.get("eps", 1e-5)),
            )

        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1
        self.num_class = args.get("num_class", 1)
        self.visualize_feature_flag = args.get("visualize_feature", False)

        # Fusion network building
        fusion_map = {
            "max": MaxFusion,
            "att": lambda: AttFusion(args["att"]["feat_dim"]),
            "disconet": lambda: DiscoFusion(args["disconet"]["feat_dim"]),
            "v2vnet": lambda: V2VNetFusion(args["v2vnet"]),
            "v2xvit": lambda: V2XViTFusion(args["v2xvit"]),
            "cobevt": lambda: CoBEVT(args["cobevt"]),
            "where2comm": lambda: Where2commFusion(args["where2comm"]),
            "who2com": lambda: Who2comFusion(args["who2com"]),
        }
        self.fusion_net = fusion_map[args["fusion_method"]]() if args["fusion_method"] != "max" else MaxFusion()

        # Shared Heads
        self.cls_head = nn.Conv2d(
            args["in_head"],
            args["anchor_number"] * self.num_class * self.num_class,
            kernel_size=1,
        )
        self.reg_head = nn.Conv2d(
            args["in_head"],
            7 * args["anchor_number"] * self.num_class,
            kernel_size=1,
        )
        self.dir_head = nn.Conv2d(
            args["in_head"],
            args["dir_args"]["num_bins"] * args["anchor_number"],
            kernel_size=1,
        )

        # Initialize parameter freezing
        self.model_train_init_adapter()
        check_trainable_module(self)

    def model_train_init_adapter(self):
        # Freeze all modules first.
        for p in self.parameters():
            p.requires_grad_(False)
        self.apply(fix_bn)

        # Only unfreeze domain adapter branch.
        for source_modality in self.source_modalities:
            adapter_module = self.domain_adapters[source_modality]
            for p in adapter_module.parameters():
                p.requires_grad_(True)
            adapter_module.apply(self._unfix_bn)

    def _build_scene_aligned_pairs(
        self,
        agent_modality_list: List[str],
        record_len: torch.Tensor,
        source_modality: str,
        target_modality: str,
    ) -> Tuple[List[int], List[int]]:
        """
        Build scene-wise source-target feature index pairs.

        Parameters
        ----------
        agent_modality_list : List[str]
            Flat modality list of all agents in batch, length N_agent_total.
        record_len : torch.Tensor
            Number of agents per scene, shape [B_scene].
        source_modality : str
            Source modality name.
        target_modality : str
            Target modality name.

        Returns
        -------
        source_feature_indices : List[int]
            Local indices in `modality_feature_dict[source_modality]`.
        target_feature_indices : List[int]
            Local indices in `modality_feature_dict[target_modality]`, matched
            scene-by-scene to source indices.
        """
        modality_occurrence_counter = {m: 0 for m in self.modality_name_list}
        source_feature_indices: List[int] = []
        target_feature_indices: List[int] = []

        cursor = 0
        for scene_len in record_len.tolist():
            scene_source_indices: List[int] = []
            scene_target_indices: List[int] = []

            for _ in range(int(scene_len)):
                modality_name = agent_modality_list[cursor]
                if modality_name not in modality_occurrence_counter:
                    modality_occurrence_counter[modality_name] = 0
                occurrence_idx = modality_occurrence_counter[modality_name]
                modality_occurrence_counter[modality_name] = occurrence_idx + 1

                if modality_name == source_modality:
                    scene_source_indices.append(occurrence_idx)
                if modality_name == target_modality:
                    scene_target_indices.append(occurrence_idx)
                cursor += 1

            if len(scene_source_indices) == 0 or len(scene_target_indices) == 0:
                continue

            # One reference target per scene; all source agents in this scene align to it.
            scene_target_ref = scene_target_indices[0]
            source_feature_indices.extend(scene_source_indices)
            target_feature_indices.extend([scene_target_ref] * len(scene_source_indices))

        return source_feature_indices, target_feature_indices

    def _forward_domain_adapter(
        self,
        modality_feature_dict: Dict[str, torch.Tensor],
        output_dict: Dict,
        agent_modality_list: List[str],
        record_len: torch.Tensor,
    ) -> None:
        """
        Run domain adaptation on source modalities using scene-matched target features.

        Input
        -----
        modality_feature_dict[modality] : torch.Tensor
            Encoded BEV features of each modality, shape [N_modality, C, H, W].
        agent_modality_list : List[str]
            Flat list for all agents in batch.
        record_len : torch.Tensor
            Scene split sizes, shape [B_scene].

        Output (written into output_dict)
        -------------------------------
        adapter_struct_source : [N_source_total, C, H, W]
        adapter_feat_source : [N_source_total, C, H, W]
        adapter_feat_target : [N_source_total, C, H, W]
        adapter_mask_source : [N_source_total, 1, H, W]
        adapter_mask_target : [N_source_total, 1, H, W]
        adapter_mask_logits : [N_source_total, 1, H, W]
        adapter_cls_preds / reg_preds / dir_preds : detection outputs on adapted source features
        teacher_cls_preds / reg_preds / dir_preds : target-head outputs for consistency supervision
        """
        if self.target_modality not in modality_feature_dict:
            return

        target_feat_all = modality_feature_dict[self.target_modality]
        if target_feat_all.shape[0] == 0:
            return

        aggregated_output = {
            "adapter_struct_source": [],
            "adapter_feat_source": [],
            "adapter_feat_target": [],
            "adapter_mask_source": [],
            "adapter_mask_target": [],
            "adapter_mask_logits": [],
            "adapter_cls_preds": [],
            "adapter_reg_preds": [],
            "adapter_dir_preds": [],
            "teacher_cls_preds": [],
            "teacher_reg_preds": [],
            "teacher_dir_preds": [],
        }

        for source_modality in self.source_modalities:
            if source_modality not in modality_feature_dict:
                continue

            source_feat_all = modality_feature_dict[source_modality]
            if source_feat_all.shape[0] == 0:
                continue

            source_indices, target_indices = self._build_scene_aligned_pairs(
                agent_modality_list=agent_modality_list,
                record_len=record_len,
                source_modality=source_modality,
                target_modality=self.target_modality,
            )
            if len(source_indices) == 0:
                continue

            source_indices_tensor = torch.as_tensor(source_indices, device=source_feat_all.device, dtype=torch.long)
            target_indices_tensor = torch.as_tensor(target_indices, device=target_feat_all.device, dtype=torch.long)

            # source_feat_batch / target_feat_batch: [N_pair, C, H, W]
            source_feat_batch = source_feat_all.index_select(0, source_indices_tensor)
            target_feat_batch = target_feat_all.index_select(0, target_indices_tensor)
            target_feat_for_adapter = target_feat_batch.detach() if self.detach_target_feature else target_feat_batch

            adapter_out = self.domain_adapters[source_modality](
                source_feat=source_feat_batch,
                target_feat=target_feat_for_adapter,
            )
            adapted_source_batch = adapter_out["adapted_source_feat"]
            adapted_source_batch = adapted_source_batch.to(dtype=source_feat_all.dtype)

            source_feat_updated = source_feat_all.clone()
            source_feat_updated.index_copy_(0, source_indices_tensor, adapted_source_batch)
            modality_feature_dict[source_modality] = source_feat_updated

            adapter_cls_preds = self.cls_head(adapted_source_batch)
            adapter_reg_preds = self.reg_head(adapted_source_batch)
            adapter_dir_preds = self.dir_head(adapted_source_batch)

            with torch.no_grad():
                teacher_cls_preds = self.cls_head(target_feat_batch)
                teacher_reg_preds = self.reg_head(target_feat_batch)
                teacher_dir_preds = self.dir_head(target_feat_batch)

            aggregated_output["adapter_struct_source"].append(adapter_out["structural_source_feat"])
            aggregated_output["adapter_feat_source"].append(adapted_source_batch)
            aggregated_output["adapter_feat_target"].append(adapter_out["reference_target_feat"])
            aggregated_output["adapter_mask_source"].append(adapter_out["source_fg_mask"])
            aggregated_output["adapter_mask_target"].append(adapter_out["target_fg_mask"])
            aggregated_output["adapter_mask_logits"].append(adapter_out["source_fg_logits"])
            aggregated_output["adapter_cls_preds"].append(adapter_cls_preds)
            aggregated_output["adapter_reg_preds"].append(adapter_reg_preds)
            aggregated_output["adapter_dir_preds"].append(adapter_dir_preds)
            aggregated_output["teacher_cls_preds"].append(teacher_cls_preds)
            aggregated_output["teacher_reg_preds"].append(teacher_reg_preds)
            aggregated_output["teacher_dir_preds"].append(teacher_dir_preds)

        if len(aggregated_output["adapter_feat_source"]) == 0:
            return

        output_dict.update({k: torch.cat(v, dim=0) for k, v in aggregated_output.items()})

    def forward(self, data_dict):
        output_dict = {}
        agent_modality_list = data_dict["agent_modality_list"]
        affine_matrix = normalize_pairwise_tfm(data_dict["pairwise_t_matrix"], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict["record_len"]

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        # 1. Feature encoding.
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue

            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            if not isinstance(eval(f"self.backbone_{modality_name}"), nn.Identity):
                feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})["spatial_features_2d"]
            feature = eval(f"self.shrinker_{modality_name}")(feature)

            modality_feature_dict[modality_name] = feature

        # 2. Crop camera features to common visible range.
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H * eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W * eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)

        # 3. Domain adaptation branch (source -> target semantic space).
        self._forward_domain_adapter(
            modality_feature_dict=modality_feature_dict,
            output_dict=output_dict,
            agent_modality_list=agent_modality_list,
            record_len=record_len,
        )

        # 4. Assemble heterogeneous features.
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        # 5. Feature fusion.
        fused_feature = self.fusion_net(heter_feature_2d, record_len, affine_matrix)

        if self.visualize_feature_flag:
            output_dict["heter_feature"] = heter_feature_2d
            output_dict["fused_feature"] = fused_feature

        # 6. Shared detection heads.
        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({"cls_preds": cls_preds, "reg_preds": reg_preds, "dir_preds": dir_preds})

        return output_dict