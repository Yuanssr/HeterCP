import numpy as np
import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F

class Box2FeatureGenerator(nn.Module):
    """
    将 {3D box, score} 列表映射为稠密 BEV 特征图。
    输入: box_list: list[batch]，box_list[i] 是非 ego 车辆的列表
           每个元素是 dict: pred_box_tensor [Ni, 8, 3], pred_score [Ni]
    输出: feature_list: list[batch]，feature_list[i] -> [non_ego_num, C, H, W]
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.pc_range = model_cfg["pc_range"]

        # 分辨率: 单位米 -> 网格索引
        self.voxel_size = model_cfg["grid_size"]  # [dx, dy]
        dx, dy, dz = self.voxel_size
        x_size = (self.pc_range[3] - self.pc_range[0]) / dx
        y_size = (self.pc_range[4] - self.pc_range[1]) / dy
        self.H = int(round(y_size))  # 注意: y 映射到行，高度
        self.W = int(round(x_size))  # x 映射到列，宽度

        self.embed_dim = model_cfg.get("embed_dim", 256)

        # 3 层 MLP: 输入 8*3 + 1(score) = 25
        self.mlp = nn.Sequential(
            nn.Linear(25, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # 3~4 个 ResBlock 做空间扩散
        def make_block(c):
            return nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
            )
        self.refine_blocks = nn.ModuleList([make_block(self.embed_dim) for _ in range(3)])

    def forward(self, box_list, device, target_hw=None):
        """
        box_list: 外层 len = batch，内层 len = non_ego_num
        返回: feature_list，同样的结构，feature_list[i] -> [non_ego_num, C, H, W]
        """
        feature_list = []
        for scene_boxes in box_list:  # 遍历 batch
            cav_feats = []
            for cav_boxes in scene_boxes:  # 遍历非 ego 车辆
                pred_box = cav_boxes["pred_box_tensor"]  # [N, 8, 3] or empty
                pred_score = cav_boxes["pred_score"]     # [N]
                if pred_box is None or pred_box.numel() == 0:
                    # 空场景，直接零图
                    cav_feats.append(torch.zeros(self.embed_dim, self.H, self.W, device=device))
                    continue

                
                N = pred_box.shape[0]

                # --- 对象编码: 每个框 8*3 点展平 + 分数 ---
                geom_feat = pred_box.reshape(N, -1)  # [N, 24]
                obj_feat = torch.cat([geom_feat, pred_score.unsqueeze(1)], dim=1)  # [N, 25]
                obj_feat = self.mlp(obj_feat)  # [N, C]
                obj_feat = obj_feat * pred_score.unsqueeze(1)  # 置信度加权

                # --- BEV 投影 ---
                # 将 3D 角点 (x,y,z) -> BEV 网格索引:
                # gx = floor((x - xmin) / dx), gy = floor((y - ymin) / dy)
                corners = pred_box[..., :2]  # [N, 8, 2]
                xmin, ymin, _, xmax, ymax, _ = self.pc_range
                dx, dy, dz = self.voxel_size
                gx = torch.floor((corners[..., 0] - xmin) / dx)  # [N, 8]
                gy = torch.floor((corners[..., 1] - ymin) / dy)  # [N, 8]

                # 求覆盖矩形范围
                gx_min = gx.min(dim=1).values.clamp(0, self.W - 1)
                gx_max = gx.max(dim=1).values.clamp(0, self.W - 1)
                gy_min = gy.min(dim=1).values.clamp(0, self.H - 1)
                gy_max = gy.max(dim=1).values.clamp(0, self.H - 1)

                # --- 将 obj_feat 填充到矩形区域 ---
                feat_map = torch.zeros(self.embed_dim, self.H, self.W).to(device)
                for i in range(N):
                    x0, x1 = int(gx_min[i].item()), int(gx_max[i].item())
                    y0, y1 = int(gy_min[i].item()), int(gy_max[i].item())
                    if x1 < x0 or y1 < y0:
                        continue
                    # 区域填充: 直接复制向量到该区域
                    feat_map[:, y0:y1 + 1, x0:x1 + 1] = obj_feat[i].view(-1, 1, 1)

                # --- 卷积补全 (ResBlocks) ---
                x = feat_map.unsqueeze(0)  # [1, C, H, W]
                for block in self.refine_blocks:
                    residual = x
                    x = block(x)
                    x = F.relu(x + residual, inplace=True)
                cav_feats.append(x.squeeze(0))  # [C, H, W]

            # 堆叠当前场景所有非 ego 车的特征: [non_ego_num, C, H, W]
            if len(cav_feats) == 0:
                # 返回空占位，分辨率与 target_hw 对齐
                    feature_list.append(torch.zeros(0, self.embed_dim, self.H, self.W, device=device))
            else:
                feature_list.append(torch.stack(cav_feats, dim=0))
        return feature_list



class Box2FeatureGeneratorV2(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.pc_range = model_cfg["pc_range"]

        # 分辨率: 单位米 -> 网格索引
        self.voxel_size = model_cfg["grid_size"]  # [dx, dy]
        dx, dy, dz = self.voxel_size
        x_size = (self.pc_range[3] - self.pc_range[0]) / dx
        y_size = (self.pc_range[4] - self.pc_range[1]) / dy
        self.H = int(round(y_size))  # 注意: y 映射到行，高度
        self.W = int(round(x_size))  # x 映射到列，宽度

        self.embed_dim = model_cfg.get("embed_dim", 256)

        # 3 层 MLP: 输入 8*3 + 1(score) = 25
        self.mlp = nn.Sequential(
            nn.Linear(25, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # 3~4 个 ResBlock 做空间扩散
        def make_block(c):
            return nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
            )
        self.refine_blocks = nn.ModuleList([make_block(self.embed_dim) for _ in range(3)])
        
    def forward(self, box_list, device, target_hw=None):
        feature_list = []
        for scene_boxes in box_list:  # 遍历 batch
            cav_feats = []
            for cav_boxes in scene_boxes:  # 遍历非 ego 车辆
                pred_box = cav_boxes["pred_box_tensor"]  # [N, 8, 3] or empty
                pred_score = cav_boxes["pred_score"]     # [N]
                if pred_box is None or pred_box.numel() == 0:
                    cav_feats.append(torch.zeros(self.embed_dim, self.H, self.W, device=device))
                    continue

                N = pred_box.shape[0]
                # 对象编码
                geom_feat = pred_box.reshape(N, -1)
                obj_feat = torch.cat([geom_feat, pred_score.unsqueeze(1)], dim=1)
                obj_feat = self.mlp(obj_feat) * pred_score.unsqueeze(1)  # [N, C]

                xmin, ymin, _, xmax, ymax, _ = self.pc_range
                dx, dy, dz = self.voxel_size

                feat_sum = torch.zeros(self.embed_dim, self.H, self.W, device=device)
                feat_cnt = torch.zeros(self.H, self.W, device=device)

                for i in range(N):
                    # 取底面四点 (假设前 4 个为底面角点)
                    corners_xy = pred_box[i, :4, :2]
                    # 转到栅格坐标
                    gx = (corners_xy[:, 0] - xmin) / dx
                    gy = (corners_xy[:, 1] - ymin) / dy
                    # 计算包围盒
                    x0 = torch.clamp(torch.floor(gx.min()), 0, self.W - 1).int()
                    x1 = torch.clamp(torch.ceil(gx.max()), 0, self.W - 1).int()
                    y0 = torch.clamp(torch.floor(gy.min()), 0, self.H - 1).int()
                    y1 = torch.clamp(torch.ceil(gy.max()), 0, self.H - 1).int()
                    if x1 < x0 or y1 < y0:
                        continue
                    # 网格中心坐标
                    xs = torch.arange(x0, x1 + 1, device=device, dtype=torch.float32)
                    ys = torch.arange(y0, y1 + 1, device=device, dtype=torch.float32)
                    grid_x, grid_y = torch.meshgrid(xs + 0.5, ys + 0.5, indexing='ij')  # [Wbox, Hbox]
                    pts = torch.stack([grid_x, grid_y], dim=-1)  # [..., 2]

                    poly = torch.stack([gx, gy], dim=1)  # [4,2]
                    # 逐边叉积，同向为内点
                    def cross(a, b):
                        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

                    v01 = poly[1] - poly[0]
                    v12 = poly[2] - poly[1]
                    v23 = poly[3] - poly[2]
                    v30 = poly[0] - poly[3]

                    p0 = pts - poly[0]
                    p1 = pts - poly[1]
                    p2 = pts - poly[2]
                    p3 = pts - poly[3]

                    c0 = cross(v01, p0)
                    c1 = cross(v12, p1)
                    c2 = cross(v23, p2)
                    c3 = cross(v30, p3)

                    inside = (c0 >= 0) & (c1 >= 0) & (c2 >= 0) & (c3 >= 0)
                    if not inside.any():
                        continue
                    # 累加到特征图
                    mask = inside.transpose(0, 1).unsqueeze(0)  # [1, Hbox, Wbox]
                    feat_sum[:, y0:y1 + 1, x0:x1 + 1] += obj_feat[i].view(-1, 1, 1) * mask
                    feat_cnt[y0:y1 + 1, x0:x1 + 1] += mask.squeeze(0)

                # 均值化，避免重叠覆盖
                feat_cnt = feat_cnt.clamp(min=1.0)
                feat_map = feat_sum / feat_cnt.unsqueeze(0)
                # refine
                x = feat_map.unsqueeze(0)
                for block in self.refine_blocks:
                    residual = x
                    x = block(x)
                    x = F.relu(x + residual, inplace=True)
                cav_feats.append(x.squeeze(0))

            if len(cav_feats) == 0:
                feature_list.append(torch.zeros(0, self.embed_dim, self.H, self.W, device=device))
            else:
                feature_list.append(torch.stack(cav_feats, dim=0))
        return feature_list