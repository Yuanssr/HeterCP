# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import numpy as np
import copy

from opencood.tools.inference_utils import get_cav_box
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev

def visualize(infer_result, pcd, pc_range, save_path, method='3d', left_hand=False, v2xreal_flag=False):
        """
        Visualize the prediction, ground truth with point cloud together.
        They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

        Parameters
        ----------
        infer_result:
            pred_box_tensor : torch.Tensor
                (N, 8, 3) prediction.

            gt_tensor : torch.Tensor
                (N, 8, 3) groundtruth bbx
            
            uncertainty_tensor : optional, torch.Tensor
                (N, ?)

            lidar_agent_record: optional, torch.Tensor
                (N_agnet, )


        pcd : torch.Tensor
            PointCloud, (N, 4).

        pc_range : list
            [xmin, ymin, zmin, xmax, ymax, zmax]

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        method: str, 'bev' or '3d'

        """
        plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
        pc_range = [int(i) for i in pc_range]
        pcd_np = pcd.cpu().numpy()

        pred_box_tensor = infer_result.get("pred_box_tensor", None)
        gt_box_tensor = infer_result.get("gt_box_tensor", None)

        if pred_box_tensor is not None:
            pred_box_np = pred_box_tensor.cpu().numpy()
            pred_name = ['pred'] * pred_box_np.shape[0]

            score = infer_result.get("score_tensor", None)
            if v2xreal_flag:
                score = score[:,0]  # only keep the confidence score for v2xreal
            if score is not None:
                score_np = score.cpu().numpy()
                pred_name = [f'score:{score_np[i]:.3f}' for i in range(score_np.shape[0])]

            uncertainty = infer_result.get("uncertainty_tensor", None)
            if uncertainty is not None:
                uncertainty_np = uncertainty.cpu().numpy()
                uncertainty_np = np.exp(uncertainty_np)
                d_a_square = 1.6**2 + 3.9**2
                
                if uncertainty_np.shape[1] == 3:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) 
                    # yaw angle is in radian, it's the same in g2o SE2's setting.

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:.3f} a_u:{uncertainty_np[i,2]:.3f}' \
                                    for i in range(uncertainty_np.shape[0])]

                elif uncertainty_np.shape[1] == 2:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f}' \
                                    for i in range(uncertainty_np.shape[0])]

                elif uncertainty_np.shape[1] == 7:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f} a_u:{uncertainty_np[i,6]:3f}' \
                                    for i in range(uncertainty_np.shape[0])]                    

        if gt_box_tensor is not None:
            gt_box_np = gt_box_tensor.cpu().numpy()
            gt_name = ['gt'] * gt_box_np.shape[0]

        if method == 'bev':
            canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=left_hand) 

            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
            canvas.draw_canvas_points(canvas_xy[valid_mask]) # Only draw valid points
            if gt_box_tensor is not None:
                #canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=['']*len(gt_name), box_line_thickness=4) # paper visualization
            if pred_box_tensor is not None:
                #canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=['']*len(pred_name), box_line_thickness=4) # paper visualization

            # heterogeneous
            agent_modality_list = infer_result.get("agent_modality_list", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if agent_modality_list is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                color_map = {
                    "m1": (0, 114, 178),    # 深蓝
                    "m2": (213, 94, 0),     # 朱红
                    "m3": (0, 158, 115),    # 蓝绿
                    "m4": (204, 121, 167),  # 紫红
                    "m5": (230, 159, 0),    # 橘黄
                    "m6": (86, 180, 233),   # 天蓝
                    "m7": (200, 200, 50),   # 暗黄 (避免纯黄在白底看不清)
                    "m8": (178, 34, 34),    # 砖红
                }
                for i, modality_name in enumerate(agent_modality_list):
                    color = color_map.get(modality_name, (128, 128, 128)) # 未知模态默认灰色

                    render_dict = {'m1': 'L-P', 'm2':"C-E", 'm3':'L-S', 'm4':'C-R'}
                    canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=[modality_name])
                    # canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=[render_dict[modality_name]], box_text_size=1.5, box_line_thickness=3) # paper visualization



        elif method == '3d':
            canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
            canvas.draw_canvas_points(canvas_xy[valid_mask])
            if gt_box_tensor is not None:
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
            if pred_box_tensor is not None:
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)

            # heterogeneous
            agent_modality_list = infer_result.get("agent_modality_list", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if agent_modality_list is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                color_map = {
                    "m1": (0, 114, 178),    # 深蓝
                    "m2": (213, 94, 0),     # 朱红
                    "m3": (0, 158, 115),    # 蓝绿
                    "m4": (204, 121, 167),  # 紫红
                    "m5": (230, 159, 0),    # 橘黄
                    "m6": (86, 180, 233),   # 天蓝
                    "m7": (200, 200, 50),   # 暗黄
                    "m8": (178, 34, 34),    # 砖红
                }
                for i, modality_name in enumerate(agent_modality_list):
                    color = color_map.get(modality_name, (128, 128, 128)) # 未知模态默认灰色
                    canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=[modality_name])

        else:
            raise(f"Not Completed for f{method} visualization.")

        plt.axis("off")

        plt.imshow(canvas.canvas)
        plt.tight_layout()
        plt.savefig(save_path, transparent=False, dpi=500)
        plt.clf()
        plt.close()


def visualize_feature(infer_result , index, save_path_root, channels = ''):
    """
    Visualize the feature maps.

    Parameters
    ----------
    infer_result : dict
        The output dict from the model's forward function, which should contain:
        heter_feature : torch.Tensor
            (A, C, H, W) Heterogeneous feature before fusion. A is the number of agents per frame.

        fused_feature : torch.Tensor
            (1, C, H, W) Fused feature after fusion. Batch size is 1.

        agent_modality_list : list of str
            List of modality names for each agent.

    index : int
        Index of the current sample.

    save_path_root : str
        Root path to save the visualization results.

    channels : list or tuple of int, optional
        Which channels to visualize. If None, visualize the mean over channels.

    """
    heter_feature, fused_feature, agent_modality_list = infer_result['heter_feature'], infer_result['fused_feature'], infer_result['agent_modality_list']
    
    heter_feature_np = heter_feature.cpu().numpy()
    fused_feature_np = fused_feature.cpu().numpy()

    A, C, H, W = heter_feature_np.shape

    # 需要展示的通道列表；None 时用 'mean'
    if channels is None:
        channels_to_show = ['mean']
    elif channels == 'all':
        channels_to_show = list(range(C))
    else:
        channels_to_show = [c for c in channels if isinstance(c, int) and 0 <= c < C]
        if not channels_to_show:
            channels_to_show = ['mean']

    for n in range(A):
        for ch in channels_to_show:
            if ch == 'mean':
                img = np.mean(heter_feature_np[n], axis=0)
                suffix = "mean"
            else:
                img = heter_feature_np[n, ch]
                suffix = f"c{ch:03d}"
            heter_save_path = f"{save_path_root}/heter_{index:05d}_agent{n+1}_{agent_modality_list[n]}_{suffix}.png"

            plt.figure(figsize=[W/10, H/10])
            plt.axis("off")
            plt.imshow(img,cmap = 'hot')
            plt.tight_layout()
            plt.savefig(heter_save_path, transparent=False, dpi=100)
            plt.clf()
            plt.close()

    for ch in channels_to_show:
        if ch == 'mean':
            img = np.mean(fused_feature_np[0], axis=0)
            suffix = "mean"
        else:
            # fused_feature 的通道范围同样为 C
            img = fused_feature_np[0, ch]
            suffix = f"c{ch:03d}"
        fused_save_path = f"{save_path_root}/fused_{index:05d}_{suffix}.png"
        plt.figure(figsize=[W/10, H/10])
        plt.axis("off")
        plt.imshow(img,cmap = 'hot')
        plt.tight_layout()
        plt.savefig(fused_save_path, transparent=False, dpi=500)
        plt.clf()
        plt.close()


def visualize_feature_reconstruction(infer_result , index, save_path_root, channels = None):
    """
    Visualize the feature maps.

    Parameters
    ----------
    infer_result : dict
        The output dict from the model's forward function, which should contain:
        heter_feature : torch.Tensor
            (A, C, H, W) Heterogeneous feature before fusion. A is the number of agents per frame.
        heter_feature_reconstruction : torch.Tensor
            (A, C, H, W) Reconstructed heterogeneous feature after shrinking. 

        fused_feature : torch.Tensor
            (1, C, H, W) Fused feature after fusion. Batch size is 1.

        agent_modality_list : list of str
            List of modality names for each agent.

    index : int
        Index of the current sample.

    save_path_root : str
        Root path to save the visualization results.

    channels : list or tuple of int, optional
        Which channels to visualize. If None, visualize the mean over channels.

    """
    heter_feature, heter_feature_reconstruction, fused_feature, agent_modality_list = infer_result['heter_feature'], infer_result['heter_feature_reconstruction'], infer_result['fused_feature'], infer_result['agent_modality_list']
    
    heter_feature_np = heter_feature.cpu().numpy()
    heter_feature_reconstruction_np = heter_feature_reconstruction.cpu().numpy()
    fused_feature_np = fused_feature.cpu().numpy()

    A, C, H, W = heter_feature_np.shape

    # 需要展示的通道列表；None 时用 'mean'
    if channels is None:
        channels_to_show = ['mean']
    elif channels == 'all':
        channels_to_show = list(range(C))
    else:
        channels_to_show = [c for c in channels if isinstance(c, int) and 0 <= c < C]
        if not channels_to_show:
            channels_to_show = ['mean']

    for n in range(A):
        for ch in channels_to_show:
            if ch == 'mean':
                img = np.mean(heter_feature_np[n], axis=0)
                suffix = "mean"
            else:
                img = heter_feature_np[n, ch]
                suffix = f"c{ch:03d}"
            heter_save_path = f"{save_path_root}/heter_{index:05d}_agent{n+1}_{agent_modality_list[n]}_{suffix}.png"

            plt.figure(figsize=[W/10, H/10])
            plt.axis("off")
            plt.imshow(img)
            plt.tight_layout()
            plt.savefig(heter_save_path, transparent=False, dpi=100)
            plt.clf()
            plt.close()

    for n in range(A):
        for ch in channels_to_show:
            if ch == 'mean':
                img = np.mean(heter_feature_reconstruction_np[n], axis=0)
                suffix = "mean"
            else:
                img = heter_feature_reconstruction_np[n, ch]
                suffix = f"c{ch:03d}"
            recon_save_path = f"{save_path_root}/recon_{index:05d}_agent{n+1}_{agent_modality_list[n]}_{suffix}.png"
            plt.figure(figsize=[W/10, H/10]); plt.axis("off"); plt.imshow(img)
            plt.tight_layout(); plt.savefig(recon_save_path, transparent=False, dpi=100)
            plt.clf(); plt.close()
            
    for ch in channels_to_show:
        if ch == 'mean':
            img = np.mean(fused_feature_np[0], axis=0)
            suffix = "mean"
        else:
            # fused_feature 的通道范围同样为 C
            img = fused_feature_np[0, ch]
            suffix = f"c{ch:03d}"
        fused_save_path = f"{save_path_root}/fused_{index:05d}_{suffix}.png"
        plt.figure(figsize=[W/10, H/10])
        plt.axis("off")
        plt.imshow(img)
        plt.tight_layout()
        plt.savefig(fused_save_path, transparent=False, dpi=500)
        plt.clf()
        plt.close()