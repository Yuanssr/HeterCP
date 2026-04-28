# -*- coding: utf-8 -*-

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SpatialAlign(nn.Module):
    def __init__(self, target_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.target_size = target_size

    def forward(
        self,
        feat: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        out_size = target_size if target_size is not None else self.target_size
        if out_size is None:
            return feat
        return F.interpolate(feat, size=out_size, mode="bilinear", align_corners=False)


class HeterDomainAdapter(nn.Module):
    """
    Cross-domain heterogeneous feature adapter with foreground-aware correction.

    Stage 1:
        - channel projection (C_source -> C_target)
        - spatial alignment of source feature to target BEV size

    Stage 2:
        - foreground estimator
        - foreground-aware masked instance normalization (FA-IN)
        - depthwise residual refinement
    """

    def __init__(
        self,
        c_source: int,
        c_target: int,
        target_size: Optional[Tuple[int, int]] = None,
        alpha: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = alpha

        # Stage 1: structural alignment
        self.channel_proj = nn.Sequential(
            nn.Conv2d(c_source, c_target, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_target),
        )
        self.spatial_align = _SpatialAlign(target_size)

        # Stage 2.1: foreground estimator
        self.fg_estimator = nn.Sequential(
            nn.Conv2d(c_target, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Stage 2.3: residual refinement with depthwise separable conv
        self.refine = nn.Sequential(
            nn.Conv2d(c_target, c_target, kernel_size=3, padding=1, groups=c_target, bias=False),
            nn.BatchNorm2d(c_target),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_target, c_target, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_target),
        )

    def _align_mask(self, mask: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if mask.shape[-2:] != feat.shape[-2:]:
            mask = F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        return mask.clamp(0.0, 1.0)

    def masked_instance_norm(self, feat: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute weighted statistics under a soft spatial mask.

        Parameters
        ----------
        feat : torch.Tensor
            Feature map, shape [B, C, H, W].
        mask : torch.Tensor
            Soft mask, shape [B, 1, H, W] (or broadcastable spatial size).

        Returns
        -------
        mu : torch.Tensor
            Masked channel mean, shape [B, C, 1, 1].
        sigma : torch.Tensor
            Masked channel std, shape [B, C, 1, 1].
        """
        mask = self._align_mask(mask, feat)
        mask_expanded = mask.expand(-1, feat.shape[1], -1, -1)
        weight_sum = mask_expanded.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)

        mu = (feat * mask_expanded).sum(dim=(2, 3), keepdim=True) / weight_sum
        var = ((feat - mu) ** 2 * mask_expanded).sum(dim=(2, 3), keepdim=True) / weight_sum
        sigma = torch.sqrt(var + self.eps)
        return mu, sigma

    def forward(
        self,
        source_feat: torch.Tensor,
        target_feat: torch.Tensor,
        target_fg_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        source_feat : torch.Tensor
            Source-domain BEV feature, shape [B_s, C_source, H_s, W_s].
        target_feat : torch.Tensor
            Target-domain reference BEV feature, shape [B_s, C_target, H_t, W_t].
            Note: batch dimension should be paired with `source_feat`.
        target_fg_mask : Optional[torch.Tensor]
            Optional target foreground mask, shape [B_s, 1, H_t, W_t].

        Returns
        -------
        Dict[str, torch.Tensor]
            structural_source_feat: [B_s, C_target, H_t, W_t]
            adapted_source_feat: [B_s, C_target, H_t, W_t]
            reference_target_feat: [B_s, C_target, H_t, W_t]
            source_fg_mask: [B_s, 1, H_t, W_t]
            target_fg_mask: [B_s, 1, H_t, W_t]
            source_fg_logits: [B_s, 1, H_t, W_t]
        """
        target_size = target_feat.shape[-2:]

        # Stage 1: structural alignment
        structural_source_feat = self.channel_proj(source_feat)
        structural_source_feat = self.spatial_align(structural_source_feat, target_size)

        if target_feat.shape[1] != structural_source_feat.shape[1]:
            raise ValueError(
                f"Target channel ({target_feat.shape[1]}) does not match projected source channel "
                f"({structural_source_feat.shape[1]})."
            )

        # Stage 2.1: foreground mask estimation
        source_fg_logits = self.fg_estimator[0](structural_source_feat)
        source_fg_mask = torch.sigmoid(source_fg_logits)
        if target_fg_mask is None:
            with torch.no_grad():
                target_fg_mask = self.fg_estimator(target_feat)
        else:
            target_fg_mask = self._align_mask(target_fg_mask, target_feat)

        # Stage 2.2: foreground-aware masked IN
        source_bg_mask = 1.0 - source_fg_mask
        target_bg_mask = 1.0 - target_fg_mask

        mu_source_fg, sigma_source_fg = self.masked_instance_norm(structural_source_feat, source_fg_mask)
        mu_source_bg, sigma_source_bg = self.masked_instance_norm(structural_source_feat, source_bg_mask)
        mu_target_fg, sigma_target_fg = self.masked_instance_norm(target_feat, target_fg_mask)
        mu_target_bg, sigma_target_bg = self.masked_instance_norm(target_feat, target_bg_mask)

        source_fg_aligned = ((structural_source_feat - mu_source_fg) / sigma_source_fg) * sigma_target_fg + mu_target_fg
        source_bg_aligned = ((structural_source_feat - mu_source_bg) / sigma_source_bg) * sigma_target_bg + mu_target_bg
        source_bg_preserved = (1.0 - self.alpha) * structural_source_feat + self.alpha * source_bg_aligned

        source_fg_mask_expanded = source_fg_mask.expand(-1, structural_source_feat.shape[1], -1, -1)
        corrected_source_feat = source_fg_mask_expanded * source_fg_aligned + (1.0 - source_fg_mask_expanded) * source_bg_preserved

        # Stage 2.3: residual refinement
        adapted_source_feat = corrected_source_feat + self.refine(corrected_source_feat)

        return {
            "structural_source_feat": structural_source_feat,
            "adapted_source_feat": adapted_source_feat,
            "reference_target_feat": target_feat,
            "source_fg_mask": source_fg_mask,
            "target_fg_mask": target_fg_mask.detach(),
            "source_fg_logits": source_fg_logits,
        }
