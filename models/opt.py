"""
Created by tye at 2021/10/8
"""

import torch
import numpy as np
from models.backbone.sparseConv import SparseConvBackbone

from models.head.xcorr import PointVoxelXCorr, BoxAwareXCorr, OTXCorr
from models import base_model
import torch.nn.functional as F
from datasets import points_utils
from pointnet2.utils import pytorch_utils as pt_utils
import MinkowskiEngine as ME
from utils.metrics import estimateOverlap, estimateAccuracy
from utils.box_coder import PointResidualCoder
from utils.loss_utils import SigmoidFocalClassificationLoss

from models.head.rpn import VoxelRPN
from utils.loss_utils import FocalLoss, RegL1Loss, _sigmoid
from utils.box_coder import mot_decode


class OPT(base_model.BaseModel):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()
        self.num_sampled_point = self.config.num_sampled_points

        config = self.cal_params_by_config(config)
        self.voxel_size = torch.from_numpy(config.voxel_size).float()
        self.voxel_area = config.voxel_area
        self.scene_ground = torch.from_numpy(config.scene_ground).float()
        self.min_img_coord = torch.from_numpy(config.min_img_coord).float()
        self.xy_size = torch.from_numpy(config.xy_size).float()
        self.mode = not config.test

        if self.config.optimizer == 'sam':
            self.automatic_optimization = False

        self.backbone = SparseConvBackbone(input_feature_dim=self.config.input_channel,
                                           output_feature_dim=self.config.feature_channel,
                                           conv1_kernel_size=self.config.conv1_kernel_size,
                                           checkpoint=self.config.checkpoint_pretrained)

        self.mlp_bc = (pt_utils.Seq(3 + self.config.feature_channel)
                       .conv1d(self.config.feature_channel, bn=True)
                       .conv1d(self.config.feature_channel, bn=True)
                       .conv1d(self.config.bc_channel, activation=None))

        self.xcorr = OTXCorr(solver_iter=50, knn=32)

        self.mlp_seg = (pt_utils.Seq(self.config.out_channel)
                        .conv1d(self.config.feature_channel, bn=True)
                        .conv1d(self.config.feature_channel, bn=True)
                        .conv1d(1, activation=None))

        self.rpn = VoxelRPN(self.voxel_area, self.scene_ground, self.mode, self.voxel_size, feat_dim=36)

        self.focal_loss = FocalLoss().cuda()
        self.l1_loss_loc = RegL1Loss().cuda()
        self.l1_loss_z = RegL1Loss().cuda()

    def prepare_input(self, template_pc, search_pc, template_box, *args, **kwargs):

        template_points, idx_t = points_utils.regularize_pc(template_pc.points.T,
                                                            self.config.template_size,
                                                            seed=1)
        search_points, idx_s = points_utils.regularize_pc(search_pc.points.T,
                                                          self.config.search_size,
                                                          seed=1)

        # prepare sparse coordinates and features for search area
        search_sparse_coords, search_sparse_feat, search_sparse_idx \
            = ME.utils.sparse_quantize(search_points, search_points, return_index=True,
                                       quantization_size=self.config.sparse_voxel_size)
        if len(search_sparse_feat.shape) == 1:
            search_sparse_feat = search_sparse_feat[np.newaxis, :]
        search_sparse_coords, search_sparse_feat = ME.utils.sparse_collate([search_sparse_coords], [search_sparse_feat])

        # prepare sparse coordinates and features for template points
        tmpl_sparse_coords, tmpl_sparse_feat, tmpl_sparse_idx \
            = ME.utils.sparse_quantize(template_points, template_points, return_index=True,
                                       quantization_size=self.config.sparse_voxel_size)
        if len(tmpl_sparse_feat.shape) == 1:
            tmpl_sparse_feat = tmpl_sparse_feat[np.newaxis, :]

        tmpl_sparse_coords, tmpl_sparse_feat = ME.utils.sparse_collate([tmpl_sparse_coords], [tmpl_sparse_feat])

        template_points_torch = torch.tensor(template_points, device=self.device, dtype=torch.float32)
        search_points_torch = torch.tensor(search_points, device=self.device, dtype=torch.float32)
        template_bc = points_utils.get_point_to_box_distance(template_points, template_box)
        template_bc_torch = torch.tensor(template_bc, device=self.device, dtype=torch.float32)

        data_dict = {
            'template_points': template_points_torch[None, ...],
            't_voxel_coords': tmpl_sparse_coords,
            't_voxel_feats': tmpl_sparse_feat,
            't_voxel_inds': tmpl_sparse_idx,
            'search_points': search_points_torch[None, ...],
            's_voxel_coords': search_sparse_coords,
            's_voxel_feats': search_sparse_feat,
            's_voxel_inds': search_sparse_idx,
            'points2cc_dist_t': template_bc_torch[None, ...]
        }
        return data_dict

    def compute_loss(self, data, output):
        search_bc = data['points2cc_dist_s']
        seg_label = data['seg_label']
        sample_idxs = output['sample_idxs']  # B,N
        N = sample_idxs.shape[1]
        seg_label = seg_label.gather(dim=1, index=sample_idxs[:, :N].long())  # B,N
        search_bc = search_bc.gather(dim=1, index=sample_idxs[:, :N, None].repeat(1, 1, self.config.bc_channel).long())

        loss_reg_hm = self.focal_loss(_sigmoid(output['pred_hm']), data['hot_map'])
        loss_reg_loc = self.l1_loss_loc(output['pred_loc'], data['index_offsets'], data['local_offsets'])
        loss_reg_z = self.l1_loss_z(output['pred_z_axis'], data['index_center'], data['z_axis'])

        self.logger.experiment.add_images('hot_map_label', data['hot_map'][0:4], self.global_step)
        self.logger.experiment.add_images('hot_map_pred', output['pred_hm'][0:4], self.global_step)

        pred_search_bc = output['pred_search_bc']
        loss_bc = F.smooth_l1_loss(pred_search_bc, search_bc, reduction='none')
        loss_bc = torch.sum(loss_bc.mean(2) * seg_label) / (seg_label.sum() + 1e-6)

        loss_seg = F.binary_cross_entropy_with_logits(output['pred_seg'], seg_label)  # B, N

        out_dict = {
            "loss_reg_hm": loss_reg_hm,
            "loss_reg_loc": loss_reg_loc,
            "loss_reg_z": loss_reg_z,
            "loss_bc": loss_bc,
            "loss_seg": loss_seg
        }

        return out_dict

    def forward(self, input_dict):
        """
        :param input_dict:
        {
        'template_points': template_points.astype('float32'),
        'search_points': search_points.astype('float32'),
        'box_label': np.array(search_bbox_reg).astype('float32'),
        'bbox_size': search_box.wlh,
        'seg_label': seg_label.astype('float32'),
        'points2cc_dist_t': template_bc,
        'points2cc_dist_s': search_bc,
        }

        :return:
        """

        template = input_dict['template_points']
        search = input_dict['search_points']
        template_bc = input_dict['points2cc_dist_t']

        t_voxel_coords, t_voxel_feats, t_voxel_inds \
            = input_dict['t_voxel_coords'], input_dict['t_voxel_feats'], input_dict['t_voxel_inds']
        s_voxel_coords, s_voxel_feats, s_voxel_inds \
            = input_dict['s_voxel_coords'], input_dict['s_voxel_feats'], input_dict['s_voxel_inds']

        # backbone, return # [B, N, 3], [B, C, N], [B, N]
        template_xyz, template_feature, sample_idxs_t, _ \
            = self.backbone(template, t_voxel_coords,
                            t_voxel_feats, t_voxel_inds,
                            self.num_sampled_point // 2, self.config.use_feat_norm)
        search_xyz, search_feature, sample_idxs, search_coords \
            = self.backbone(search, s_voxel_coords,
                            s_voxel_feats, s_voxel_inds,
                            self.num_sampled_point, self.config.use_feat_norm)

        # prepare bc
        pred_search_bc = self.mlp_bc(torch.cat([search_xyz.transpose(1, 2), search_feature], dim=1))  # [B, 9, 128]
        pred_search_bc = pred_search_bc.transpose(1, 2)
        sample_idxs_t = sample_idxs_t[:, :, None]
        template_bc = template_bc.gather(dim=1, index=sample_idxs_t.repeat(1, 1, template_bc.size(2)).long())

        fusion_feature = self.xcorr(search_feature, template_feature,
                                    search_xyz, template_xyz,
                                    pred_search_bc, template_bc) # ot

        # for decorrelation
        fusion_feature_avg = F.avg_pool1d(fusion_feature, sample_idxs.size(1)).squeeze(2)  # B, C,

        pred_seg = self.mlp_seg(fusion_feature).squeeze(1)

        # proposal generation
        fusion_xyz_feature = torch.cat((search_xyz.transpose(1, 2).contiguous(),
                                        pred_seg.sigmoid().unsqueeze(1),
                                        fusion_feature), dim=1)
        pred_hm, pred_loc, pred_z_axis = self.rpn(fusion_xyz_feature, search_xyz)

        end_points = {"pred_hm": pred_hm,
                      "pred_loc": pred_loc,
                      "pred_z_axis": pred_z_axis,
                      "pred_seg": pred_seg,
                      'sample_idxs': sample_idxs,
                      "pred_search_bc": pred_search_bc,
                      "fusion_avg": fusion_feature_avg
                      }

        return end_points

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        """
        {"estimation_boxes": estimation_boxs.transpose(1, 2).contiguous(),
                  "vote_center": vote_xyz,
                  "pred_seg_score": estimation_cla,
                  "center_xyz": center_xyzs,
                  "seed_idxs":
                  "seg_label"
                  "pred_search_bc": pred_search_bc
        }
        """
        if self.global_step % 100 == 0:
            torch.cuda.empty_cache()

        # compute loss
        end_points = self(batch)
        loss_dict = self.compute_loss(batch, end_points)
        loss = loss_dict['loss_reg_hm'] * self.config.hm_weight \
               + loss_dict['loss_reg_loc'] * self.config.loc_weight \
               + loss_dict['loss_reg_z'] * self.config.z_weight \
               + loss_dict['loss_bc'] * self.config.bc_weight \
               + loss_dict['loss_seg'] * self.config.seg_weight

        # log
        self.log('loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_hm/train', loss_dict['loss_reg_hm'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_loc/train', loss_dict['loss_reg_loc'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_zaxis/train', loss_dict['loss_reg_z'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_bc/train', loss_dict['loss_bc'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_seg/train', loss_dict['loss_seg'].item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)

        self.logger.experiment.add_scalars('loss', {'loss_total': loss.item(),
                                                    'loss_reg_hm': loss_dict['loss_reg_hm'].item(),
                                                    'loss_reg_loc': loss_dict['loss_reg_loc'].item(),
                                                    'loss_reg_z': loss_dict['loss_reg_z'].item(),
                                                    'loss_bc': loss_dict['loss_bc'].item(),
                                                    'loss_seg': loss_dict['loss_seg'].item(),
                                                    },
                                           global_step=self.global_step)

        return loss

    def generate_search_area(self, sequence, current_frame_id, results_bbs):
        """
        generate search area for evaluating.

        :param sequence:
        :param current_frame_id:
        :param results_bbs:
        :return:
        """
        this_bb = sequence[current_frame_id]["3d_bbox"]
        this_pc = sequence[current_frame_id]["pc"]
        if "previous_result".upper() in self.config.reference_BB.upper():
            ref_bb = results_bbs[-1]
        elif "previous_gt".upper() in self.config.reference_BB.upper():
            previous_bb = sequence[current_frame_id - 1]["3d_bbox"]
            ref_bb = previous_bb
        elif "current_gt".upper() in self.config.reference_BB.upper():
            ref_bb = this_bb
        area_extents = torch.tensor(self.config.area_extents).reshape(3, 2).numpy()
        search_pc_crop = points_utils.generate_subwindow(this_pc, ref_bb,
                                                         scale=self.config.search_bb_scale,
                                                         offset=self.config.search_bb_offset,
                                                         limit_area=area_extents)
        return search_pc_crop, ref_bb

    def evaluate_one_sequence(self, sequence):
        """

        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []
        results_bbs = []
        for frame_id in range(len(sequence)):  # tracklet
            this_bb = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
            else:

                # preparing search area
                search_pc_crop, ref_bb = self.generate_search_area(sequence, frame_id, results_bbs)

                # update template
                template_pc, canonical_box = self.generate_template(sequence, frame_id, results_bbs)

                # construct input dict
                data_dict = self.prepare_input(template_pc, search_pc_crop, canonical_box)

                # forward
                end_points = self(data_dict)

                pred_hm = end_points["pred_hm"].sigmoid_()
                pred_loc = end_points["pred_loc"]
                pred_z_axis = end_points["pred_z_axis"]

                xy_img_z_ry = mot_decode(pred_hm, pred_loc, pred_z_axis, K=1)

                xy_img_z_ry_cpu = xy_img_z_ry.squeeze(0).detach().cpu().numpy()
                xy_img_z_ry_cpu[:, :2] = (xy_img_z_ry_cpu[:, :2] + self.min_img_coord.numpy()) * self.xy_size.numpy()
                estimate_box = xy_img_z_ry_cpu[0]

                candidate_box = points_utils.getOffsetBB(ref_bb, estimate_box[:4],
                                                         degrees=self.config.degrees,
                                                         use_z=self.config.use_z,
                                                         limit_box=self.config.limit_box)
                results_bbs.append(candidate_box)

            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)
        return ious, distances

    def cal_params_by_config(self, config):

        voxel_size = np.array(config.voxel_size)
        xy_voxle_size = np.array(config.xy_size) * config.downsample

        area_extents = np.array(config.area_extents).reshape(3, 2)
        xy_area_extents = np.array(config.xy_area_extents).reshape(2, 2)
        voxel_extents_transpose = area_extents.transpose()
        extents_transpose = xy_area_extents.transpose()

        scene_ground = voxel_extents_transpose[0]
        voxel_grid_size = np.ceil(voxel_extents_transpose[1] / voxel_size) - np.floor(
            voxel_extents_transpose[0] / voxel_size)
        voxel_grid_size = voxel_grid_size.astype(np.int32)

        min_img_coord = np.floor(extents_transpose[0] / xy_voxle_size)
        max_img_coord = np.ceil(extents_transpose[1] / xy_voxle_size) - 1
        img_size = ((max_img_coord - min_img_coord) + 1).astype(np.int32)  # [w, h]

        config.voxel_size = voxel_size
        config.voxel_area = voxel_grid_size
        config.scene_ground = scene_ground
        config.min_img_coord = min_img_coord
        config.xy_size = xy_voxle_size
        return config
