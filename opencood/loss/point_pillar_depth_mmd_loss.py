# -*- coding: utf-8 -*-

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from opencood.loss.point_pillar_depth_loss import PointPillarDepthLoss


class PointPillarDepthMMDLoss(PointPillarDepthLoss):
    """
    PointPillarDepthLoss + feature-alignment MMD loss.

    This class reuses PointPillarDepthLoss for detection/depth supervision,
    and only appends MMD-related logic.
    """

    def __init__(self, args: Dict):
        base_args = self._build_base_loss_args(args)
        super().__init__(base_args)

        self.eps = float(args.get("eps", 1e-6))
        self.use_adapter_detection = bool(args.get("use_adapter_detection", False))
        self.mmd_bandwidths: List[float] = [float(x) for x in args.get("mmd_bandwidths", [0.5, 1.0, 2.0])]

        default_weights = {
            "mmd": 0.1,
            "mmd_fg": 1.0,
            "mmd_bg": 0.1,
        }
        default_weights.update(args.get("weights", {}))
        self.weights = default_weights

    @staticmethod
    def _build_base_loss_args(args: Dict) -> Dict:
        base_args = copy.deepcopy(args)
        weights = args.get("weights", {})
        det_factor = float(weights.get("det", 1.0))

        base_args.setdefault("pos_cls_weight", float(args.get("pos_cls_weight", 1.0)))

        if "cls" not in base_args:
            base_args["cls"] = {
                "alpha": float(args.get("focal_alpha", 0.25)),
                "gamma": float(args.get("focal_gamma", 2.0)),
                "weight": float(weights.get("cls", 1.0)) * det_factor,
            }
        elif isinstance(base_args["cls"], dict) and "weight" in base_args["cls"]:
            base_args["cls"]["weight"] = float(base_args["cls"]["weight"]) * det_factor

        if "reg" not in base_args:
            base_args["reg"] = {
                "sigma": float(args.get("reg_sigma", 3.0)),
                "weight": float(weights.get("reg", 1.0)) * det_factor,
            }
        elif isinstance(base_args["reg"], dict) and "weight" in base_args["reg"]:
            base_args["reg"]["weight"] = float(base_args["reg"]["weight"]) * det_factor

        if "dir" in base_args and isinstance(base_args["dir"], dict) and "weight" in base_args["dir"]:
            base_args["dir"]["weight"] = float(base_args["dir"]["weight"]) * det_factor

        if "iou" in base_args and isinstance(base_args["iou"], dict) and "weight" in base_args["iou"]:
            base_args["iou"]["weight"] = float(base_args["iou"]["weight"]) * det_factor

        if "depth" not in base_args:
            # Keep depth path compatible even when no depth supervision is provided.
            base_args["depth"] = {"weight": float(weights.get("depth", 0.0))}

        return base_args

    @staticmethod
    def _to_float(value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.item())
        return float(value)

    def _zero_tensor(self, output_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        for value in output_dict.values():
            if isinstance(value, torch.Tensor):
                return value.new_tensor(0.0)
        return torch.tensor(0.0)

    def _resolve_detection_output(self, output_dict: Dict[str, torch.Tensor], suffix: str) -> Dict[str, torch.Tensor]:
        if not self.use_adapter_detection:
            return output_dict

        adapter_cls_key = f"adapter_cls_preds{suffix}"
        if adapter_cls_key not in output_dict:
            return output_dict

        resolved = dict(output_dict)
        resolved[f"cls_preds{suffix}"] = output_dict[adapter_cls_key]

        adapter_reg_key = f"adapter_reg_preds{suffix}"
        if adapter_reg_key in output_dict:
            resolved[f"reg_preds{suffix}"] = output_dict[adapter_reg_key]

        adapter_dir_key = f"adapter_dir_preds{suffix}"
        if adapter_dir_key in output_dict:
            resolved[f"dir_preds{suffix}"] = output_dict[adapter_dir_key]

        return resolved

    def _masked_gap(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape[-2:] != feat.shape[-2:]:
            mask = F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.clamp(0.0, 1.0)
        denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        pooled = (feat * mask).sum(dim=(2, 3), keepdim=True) / denom
        return pooled.flatten(1)

    def _multi_scale_mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or y.numel() == 0:
            return x.new_tensor(0.0)

        xx = torch.cdist(x, x, p=2) ** 2
        yy = torch.cdist(y, y, p=2) ** 2
        xy = torch.cdist(x, y, p=2) ** 2

        mmd = x.new_tensor(0.0)
        for bandwidth in self.mmd_bandwidths:
            gamma = 1.0 / (2.0 * (bandwidth**2) + self.eps)
            k_xx = torch.exp(-gamma * xx)
            k_yy = torch.exp(-gamma * yy)
            k_xy = torch.exp(-gamma * xy)
            mmd = mmd + (k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())

        return mmd / max(len(self.mmd_bandwidths), 1)

    def _compute_fa_mmd(self, output_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Prefer adapted source feature for alignment quality measurement.
        feat_source = output_dict.get("adapter_feat_source", None)
        feat_target = output_dict.get("adapter_feat_target", None)
        mask_source = output_dict.get("adapter_mask_source", None)
        mask_target = output_dict.get("adapter_mask_target", None)

        if feat_source is None:
            feat_source = output_dict.get("adapter_struct_source", None)

        if feat_source is None or feat_target is None or mask_source is None:
            zero = self._zero_tensor(output_dict)
            return zero, zero, zero

        if mask_target is None:
            mask_target = mask_source.detach()

        feat_source_fg = self._masked_gap(feat_source, mask_source)
        feat_target_fg = self._masked_gap(feat_target, mask_target)
        feat_source_bg = self._masked_gap(feat_source, 1.0 - mask_source)
        feat_target_bg = self._masked_gap(feat_target, 1.0 - mask_target)

        mmd_fg = self._multi_scale_mmd(feat_source_fg, feat_target_fg)
        mmd_bg = self._multi_scale_mmd(feat_source_bg, feat_target_bg)
        mmd_loss = self.weights["mmd_fg"] * mmd_fg + self.weights["mmd_bg"] * mmd_bg
        return mmd_loss, mmd_fg, mmd_bg

    def forward(
        self,
        output_dict: Dict[str, torch.Tensor],
        target_dict: Dict[str, torch.Tensor],
        suffix: str = "",
    ) -> torch.Tensor:
        resolved_output = self._resolve_detection_output(output_dict, suffix)
        total_loss = super().forward(resolved_output, target_dict, suffix=suffix)

        mmd_loss, mmd_fg, mmd_bg = self._compute_fa_mmd(output_dict)
        total_loss = total_loss + self.weights["mmd"] * mmd_loss

        self.loss_dict.update(
            {
                "total_loss": total_loss.item(),
                "mmd_loss": mmd_loss.detach(),
                "mmd_fg": mmd_fg.detach(),
                "mmd_bg": mmd_bg.detach(),
            }
        )
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix="", iter=None):
        total_loss = self._to_float(self.loss_dict.get("total_loss", 0.0))
        reg_loss = self._to_float(self.loss_dict.get("reg_loss", 0.0))
        cls_loss = self._to_float(self.loss_dict.get("cls_loss", 0.0))
        dir_loss = self._to_float(self.loss_dict.get("dir_loss", 0.0))
        iou_loss = self._to_float(self.loss_dict.get("iou_loss", 0.0))
        depth_loss = self._to_float(self.loss_dict.get("depth_loss", 0.0))
        mmd_loss = self._to_float(self.loss_dict.get("mmd_loss", 0.0))

        print(
            "[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
            " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || Depth Loss: %.4f || MMD Loss: %.4f"
            % (
                epoch,
                batch_id + 1,
                batch_len,
                suffix,
                total_loss,
                cls_loss,
                reg_loss,
                dir_loss,
                iou_loss,
                depth_loss,
                mmd_loss,
            )
        )

        if writer is not None:
            step = epoch * batch_len + batch_id
            writer.add_scalar("Regression_loss" + suffix, reg_loss, step)
            writer.add_scalar("Confidence_loss" + suffix, cls_loss, step)
            writer.add_scalar("Dir_loss" + suffix, dir_loss, step)
            writer.add_scalar("Iou_loss" + suffix, iou_loss, step)
            writer.add_scalar("Depth_loss" + suffix, depth_loss, step)
            writer.add_scalar("MMD_loss" + suffix, mmd_loss, step)
