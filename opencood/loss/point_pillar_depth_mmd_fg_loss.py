# -*- coding: utf-8 -*-

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from opencood.loss.point_pillar_depth_mmd_loss import PointPillarDepthMMDLoss


class PointPillarDepthMMDFGLoss(PointPillarDepthMMDLoss):
    """
    PointPillarDepthMMDLoss + foreground-mask supervision.

    This class only adds foreground-mask logic on top of PointPillarDepthMMDLoss,
    so detection/depth/MMD behaviors are all inherited.
    """

    def __init__(self, args: Dict):
        super().__init__(args)

        self.mask_focal_alpha = float(args.get("mask_focal_alpha", 0.25))
        self.mask_focal_gamma = float(args.get("mask_focal_gamma", 2.0))

        if "fg_mask" not in self.weights:
            self.weights["fg_mask"] = 0.5

    def _sigmoid_focal_with_logits(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor],
        alpha: float,
        gamma: float,
    ) -> torch.Tensor:
        ce_loss = torch.clamp(preds, min=0.0) - preds * targets + torch.log1p(torch.exp(-torch.abs(preds)))
        probs = torch.sigmoid(preds)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        modulating = torch.pow(1.0 - p_t, gamma)
        alpha_factor = targets * alpha + (1.0 - targets) * (1.0 - alpha)
        loss = alpha_factor * modulating * ce_loss

        if weights is not None:
            while weights.dim() < loss.dim():
                weights = weights.unsqueeze(-1)
            loss = loss * weights
        return loss

    def _build_fg_mask_target(
        self,
        target_dict: Dict[str, torch.Tensor],
        target_hw: Tuple[int, int],
        ref_tensor: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Build foreground supervision mask from explicit BEV labels.

        Notes
        -----
        `object_bbx_center` / `object_bbx_mask` are object-list tensors where
        `object_bbx_mask` indicates whether each box slot is valid, not a BEV
        foreground supervision map. They are intentionally NOT used here.

        If no explicit foreground label is found, return None so the caller can
        skip foreground-mask supervision for this batch.
        """
        device = ref_tensor.device
        dtype = ref_tensor.dtype

        if "fg_bev_mask" in target_dict:
            fg = target_dict["fg_bev_mask"].to(device=device, dtype=dtype)
            if fg.dim() == 3:
                fg = fg.unsqueeze(1)
            elif fg.dim() == 4 and fg.shape[1] != 1 and fg.shape[-1] == 1:
                fg = fg.permute(0, 3, 1, 2)
            elif fg.dim() == 4 and fg.shape[1] != 1:
                fg = fg[:, :1]

            if fg.shape[-2:] != target_hw:
                fg = F.interpolate(fg, size=target_hw, mode="nearest")

            return fg.clamp(0.0, 1.0)

        return self._build_fg_mask_target_placeholder(target_dict, target_hw, ref_tensor)

    def _build_fg_mask_target_placeholder(
        self,
        target_dict: Dict[str, torch.Tensor],
        target_hw: Tuple[int, int],
        ref_tensor: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Placeholder hook for future foreground supervision construction.

        Implement this method when a reliable BEV foreground label (e.g.
        rasterized polygons / occupancy foreground map) is available.
        """
        _ = target_dict
        _ = target_hw
        _ = ref_tensor
        return None

    def _compute_fg_mask_loss(self, output_dict: Dict[str, torch.Tensor], target_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask_logits = output_dict.get("adapter_mask_logits", None)
        if mask_logits is None:
            return self._zero_tensor(output_dict)

        target_mask = self._build_fg_mask_target(target_dict, mask_logits.shape[-2:], mask_logits)
        if target_mask is None:
            return self._zero_tensor(output_dict)

        mask_loss = self._sigmoid_focal_with_logits(
            mask_logits,
            target_mask,
            weights=None,
            alpha=self.mask_focal_alpha,
            gamma=self.mask_focal_gamma,
        )
        return mask_loss.mean()

    def forward(
        self,
        output_dict: Dict[str, torch.Tensor],
        target_dict: Dict[str, torch.Tensor],
        suffix: str = "",
    ) -> torch.Tensor:
        total_loss = super().forward(output_dict, target_dict, suffix=suffix)

        fg_mask_loss = self._compute_fg_mask_loss(output_dict, target_dict)
        total_loss = total_loss + self.weights["fg_mask"] * fg_mask_loss

        self.loss_dict.update(
            {
                "total_loss": total_loss.item(),
                "fg_mask_loss": fg_mask_loss.detach(),
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
        fg_mask_loss = self._to_float(self.loss_dict.get("fg_mask_loss", 0.0))

        print(
            "[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
            " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || Depth Loss: %.4f || MMD Loss: %.4f || FG-Mask Loss: %.4f"
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
                fg_mask_loss,
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
            writer.add_scalar("FG_Mask_loss" + suffix, fg_mask_loss, step)
