from re import I
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import torch

from .anchor_head_template import AnchorHeadTemplate


class RotRegression(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, logger=None, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training, logger=logger
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_squeeze = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.fc = nn.Sequential(
            # this is hard code
            nn.Linear(grid_size[0]*grid_size[1]//4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )
        self.dropout = nn.Dropout(0.4)
        
        if self.model_cfg.get('USE_DIR_CLASSIFIER', None) is not None:
            self.dir_pred = nn.Linear(256, 1)
            self.sigmoid = nn.Sigmoid()
        else:
            self.dir_pred = None
        
        if self.model_cfg.get('USE_SHIFT_REGRESSION', None) is not None:
            self.shift_pred = nn.Linear(256, 1)
        else:
            self.shift_pred = None

        if self.model_cfg.get('USE_ROT_REGRESSION', None) is not None:
            self.rot_pred = nn.Linear(256, 1)
        else:
            self.rot_pred = None

        # self.init_weights()

    # def init_weights(self):
    #     pi = 0.01
    #     nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
    #     nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
    
    def add_sin_difference(self, rot_pred, rot_target):
        pred_embedding = torch.sin(rot_pred) * torch.cos(rot_target)
        target_embedding = torch.cos(rot_pred) * torch.sin(rot_target)
        return pred_embedding, target_embedding
    
    def get_loss(self):
        rot_preds = self.forward_ret_dict['rot_preds']
        rot_labels = self.forward_ret_dict['rot_labels']
        shift_preds = self.forward_ret_dict['shift_preds']
        shift_labels = self.forward_ret_dict['shift_labels']
        dir_preds = self.forward_ret_dict['dir_preds']
        dir_labels = self.forward_ret_dict['dir_labels']

        if dir_preds is not None:
            self.logger.info(dir_labels.data.cpu().numpy())
            dir_loss = F.binary_cross_entropy(dir_preds, dir_labels) * \
                self.model_cfg['LOSS_CONFIG']['LOSS_WEIGHTS']['dir_weight']
        else:
            dir_loss = 0

        if shift_preds is not None:
            shift_loss = F.l1_loss(shift_preds, shift_labels) * \
            self.model_cfg['LOSS_CONFIG']['LOSS_WEIGHTS']['shift_weight']
        else:
            shift_loss = 0.0

        if rot_preds is not None:
            if dir_preds is not None:
                inverse_mask = rot_labels < 0
                rot_labels[inverse_mask] = rot_labels[inverse_mask] * -1
            self.logger.info(rot_labels.data.cpu().numpy())
            pred_embedding, target_embedding = self.add_sin_difference(
                rot_preds, rot_labels
            )
            rot_loss = F.l1_loss(pred_embedding, target_embedding) * \
                self.model_cfg['LOSS_CONFIG']['LOSS_WEIGHTS']['rot_weight']
        else:
            rot_loss = 0.0
        
        rpn_loss = rot_loss + shift_loss + dir_loss
        tb_dict = {'rpn_loss': rpn_loss.item()}
        return rpn_loss, tb_dict

    def forward(self, data_dict):
        vis_gt = True
        if vis_gt:
            from pcdet.utils.visualize import draw_bev_gt, draw_bev_pts
            import numpy as np
            import cv2
            import os
            batch_size = data_dict['batch_size']
            voxel_coords = data_dict['voxel_coords']

            visualize_dir = "./visualize_rot_width_depth"
            if not os.path.exists(visualize_dir):
                os.makedirs(visualize_dir)
                print('create directory: {}'.format(visualize_dir))

            for batch_id in range(batch_size):
                frame_id = data_dict['frame_id'][batch_id]
                frame_pts_path = os.path.join(visualize_dir, 'pts_%s.png'%frame_id)
                # frame_gt_path = os.path.join(visualize_dir, 'gt_%s.png'%frame_id)

                voxel_coord = voxel_coords[voxel_coords[:,0]==batch_id][:,1:].cpu().numpy()[:, ::-1]    # (N1, 3) [x,y,z]
                # gt_boxes = data_dict["gt_boxes"][batch_id].cpu().numpy() # (K, 7)

                draw_bev_pts(frame_pts_path, voxel_coord, gt_boxes=None, area_scope = [[0, 70.4], [-40, 40], [-3, 1]], cmap_color = False, voxel_size = [0.16, 0.16, 4])
                import pdb; pdb.set_trace()
                # draw_bev_gt(frame_gt_path, voxel_coord, gt_boxes, area_scope = [[0, 70.4], [-40, 40], [-3, 1]], cmap_color = False, voxel_size = self.global_cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE'])

        spatial_features_2d = data_dict['spatial_features_2d']
        N, C, H, W = spatial_features_2d.shape
        
        # squeeze feature
        feature_squeezed = self.conv_squeeze(spatial_features_2d).view(N, -1)
        feature_fc = self.fc(feature_squeezed)
        feature_fc = self.dropout(feature_fc)
        
        if self.rot_pred is not None:
            rot_preds = self.rot_pred(feature_fc)
        else:
            rot_preds = None
        
        if self.shift_pred is not None:
            shift_preds = self.shift_pred(feature_fc)
        else:
            shift_preds = None

        if self.dir_pred is not None:
            dir_preds = self.dir_pred(feature_fc)
            dir_preds = self.sigmoid(dir_preds)
        else:
            dir_preds = None

        self.forward_ret_dict['rot_preds'] = rot_preds
        self.forward_ret_dict['shift_preds'] = shift_preds
        self.forward_ret_dict['dir_preds'] = dir_preds

        if self.training:
            self.forward_ret_dict['rot_labels'] = data_dict['rot_labels']
            self.forward_ret_dict['shift_labels'] = data_dict['shift_labels']
            self.forward_ret_dict['dir_labels'] = data_dict['dir_labels']

        if not self.training:
            data_dict['batch_rot_preds'] = rot_preds
            data_dict['batch_shift_preds'] = shift_preds
            data_dict['batch_dir_preds'] = dir_preds

        return data_dict
