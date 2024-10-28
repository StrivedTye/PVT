""" 
Created by tye at 2021/10/8
"""

import torch
import numpy as np
import os
from models.backbone.sparseConv import SparseConvBackbone

from models.head.xcorr import PointVoxelXCorr, BoxAwareXCorr, OTXCorr, P2B_XCorr
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


# class PVT(base_model.BaseModel):
#     def __init__(self, config=None, **kwargs):
#         """
#
#         :param config: input_channel, conv1_kernel_size,  refine, checkpoint,
#         :param kwargs:
#         """
#
#         super().__init__(config, **kwargs)
#         self.save_hyperparameters()
#
#         self.box_coder = PointResidualCoder(code_size=8)
#
#         self.num_sampled_point = self.config.num_sampled_points
#         self.backbone = SparseConvBackbone(input_feature_dim=self.config.input_channel,
#                                            output_feature_dim=self.config.feature_channel,
#                                            conv1_kernel_size=self.config.conv1_kernel_size,
#                                            checkpoint=self.config.checkpoint)
#         # if len(self.config.gpu) > 1:
#         #     self.backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.backbone)
#
#         self.mlp_bc = (pt_utils.Seq(3 + self.config.feature_channel)
#                        .conv1d(self.config.feature_channel, bn=True)
#                        .conv1d(self.config.feature_channel, bn=True)
#                        .conv1d(self.config.bc_channel, activation=None))
#
#         self.xcorr = PointVoxelXCorr(num_levels=self.config.num_levels,
#                                      base_scale=self.config.base_scale,
#                                      truncate_k=self.config.truncate_k,
#                                      num_knn=self.config.num_knn,
#                                      feat_channel=self.config.out_channel)
#         if self.config.use_bev:
#             self.fusion_trans = torch.nn.Sequential(
#                 ME.MinkowskiConvolution(
#                     in_channels=self.config.out_channel,
#                     out_channels=64,
#                     kernel_size=3,
#                     stride=1,
#                     dilation=1,
#                     bias=False,
#                     dimension=3), ME.MinkowskiBatchNorm(64), ME.MinkowskiReLU(),
#                 ME.MinkowskiConvolution(
#                     in_channels=64,
#                     out_channels=128,
#                     kernel_size=3,
#                     stride=1,
#                     dilation=1,
#                     dimension=3), ME.MinkowskiBatchNorm(128), ME.MinkowskiReLU(),
#             )
#
#             self.bev_rpn = BEVRPN()
#         else:
#             self.rpn = P2BVoteNetRPN2(self.config.feature_channel,
#                                       vote_channel=self.config.vote_channel,
#                                       num_proposal=self.config.num_proposal,
#                                       normalize_xyz=self.config.normalize_xyz)
#         self.focal_loss = SigmoidFocalClassificationLoss()
#
#     def prepare_input(self, template_pc, search_pc, template_box, *args, **kwargs):
#
#         template_points, idx_t = points_utils.regularize_pc(template_pc.points.T,
#                                                             self.config.template_size,
#                                                             seed=1)
#         search_points, idx_s = points_utils.regularize_pc(search_pc.points.T,
#                                                           self.config.search_size,
#                                                           seed=1)
#
#         # prepare sparse coordinates and features for search area
#         search_sparse_coords, search_sparse_feat, search_sparse_idx \
#             = ME.utils.sparse_quantize(search_points, search_points, return_index=True,
#                                        quantization_size=self.config.voxel_size)
#         if len(search_sparse_feat.shape) == 1:
#             search_sparse_feat = search_sparse_feat[np.newaxis, :]
#         search_sparse_coords, search_sparse_feat = ME.utils.sparse_collate([search_sparse_coords], [search_sparse_feat])
#
#         # prepare sparse coordinates and features for template points
#         tmpl_sparse_coords, tmpl_sparse_feat, tmpl_sparse_idx \
#             = ME.utils.sparse_quantize(template_points, template_points, return_index=True,
#                                        quantization_size=self.config.voxel_size)
#         if len(tmpl_sparse_feat.shape) == 1:
#             tmpl_sparse_feat = tmpl_sparse_feat[np.newaxis, :]
#
#         tmpl_sparse_coords, tmpl_sparse_feat = ME.utils.sparse_collate([tmpl_sparse_coords], [tmpl_sparse_feat])
#
#         template_points_torch = torch.tensor(template_points, device=self.device, dtype=torch.float32)
#         search_points_torch = torch.tensor(search_points, device=self.device, dtype=torch.float32)
#         template_bc = points_utils.get_point_to_box_distance(template_points, template_box)
#         template_bc_torch = torch.tensor(template_bc, device=self.device, dtype=torch.float32)
#
#         data_dict = {
#             'template_points': template_points_torch[None, ...],
#             't_voxel_coords': tmpl_sparse_coords,
#             't_voxel_feats': tmpl_sparse_feat,
#             't_voxel_inds': tmpl_sparse_idx,
#             'search_points': search_points_torch[None, ...],
#             's_voxel_coords': search_sparse_coords,
#             's_voxel_feats': search_sparse_feat,
#             's_voxel_inds': search_sparse_idx,
#             'points2cc_dist_t': template_bc_torch[None, ...]
#         }
#         return data_dict
#
#     def compute_loss(self, input_dict, output):
#         seg_label = input_dict['seg_label']  # B, N
#         box_search = input_dict['box_search']  # B, 7
#         box_tmpl = input_dict['box_tmpl']
#         search_bc = input_dict['points2cc_dist_s']
#
#         pred_search_bc = output['pred_search_bc']
#         center = output['center_xyz']
#         pred_box = output['estimation_boxes']
#         pred_seg = output['estimation_cla']  #[B, N]
#         pred_score = output['estimation_score']  #[B, 64]
#         vote_xyz = output['vote_xyz']
#         sample_idxs = output['sample_idxs']  # B,N
#
#         seg_label = seg_label.gather(dim=1, index=sample_idxs.long())  # B,N
#         search_bc = search_bc.gather(dim=1, index=sample_idxs[:, :, None].repeat(1, 1, self.config.bc_channel).long())
#
#         # for box reg: xt = (xg - xa)/d, dxt = log(dxg/dxa), ...[x, y, z, l, w, h, sin, cos]
#         box_label = self.box_coder.encode_torch_template_batch(gt_boxes=box_search.unsqueeze(1).repeat(1, center.size(1), 1),
#                                                                points=center,
#                                                                templates=box_tmpl.unsqueeze(1))  # [B, N, 8],
#
#         # filter out proposals that is far away from gt
#         dist = torch.sum((center - box_search[:, None, 0:3]) ** 2, dim=-1)
#         dist = torch.sqrt(dist + 1e-6)
#         objectness_label = torch.zeros_like(dist)
#         objectness_mask = torch.zeros_like(dist)
#         objectness_label[dist < 0.3] = 1
#
#         objectness_mask[dist < 0.3] = 1
#         objectness_mask[dist > 0.6] = 1
#
#         # semantic
#         # loss_seg = F.binary_cross_entropy_with_logits(pred_seg, seg_label)
#         loss_seg = self.focal_loss(pred_seg.unsqueeze(-1),
#                                    seg_label.unsqueeze(-1),
#                                    torch.ones_like(seg_label))
#         loss_seg = loss_seg.mean()
#
#         # vote
#         loss_vote = F.smooth_l1_loss(vote_xyz, box_search[:, None, :3].expand_as(vote_xyz), reduction='none')
#         loss_vote = (loss_vote.mean(2) * seg_label).sum() / (seg_label.sum() + 1e-06)
#
#         # score
#         loss_objective = F.binary_cross_entropy_with_logits(pred_score[:, :, -1], objectness_label,
#                                                             pos_weight=torch.tensor([2.0], device=self.device))
#         loss_objective = torch.sum(loss_objective * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
#
#         # box
#         loss_box = F.smooth_l1_loss(pred_box, box_label, reduction='none')
#         loss_box = torch.sum(loss_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)
#
#         # box cloud loss
#         loss_bc = F.smooth_l1_loss(pred_search_bc, search_bc, reduction='none')
#         loss_bc = torch.sum(loss_bc.mean(2) * seg_label) / (seg_label.sum() + 1e-6)
#
#         out_dict = {
#             "loss_objective": loss_objective,
#             "loss_box": loss_box,
#             "loss_seg": loss_seg,
#             "loss_vote": loss_vote,
#             "loss_bc": loss_bc
#         }
#
#         return out_dict
#
#     def compute_loss_bev(self, input_dict, output):
#         template = input_dict['template_points']
#         search = input_dict['search_points']
#         seg_label = input_dict['seg_label']  # B, N
#         box_label = input_dict['box_label']  # B, 4
#         box_size = input_dict['bbox_size']
#         search_bc = input_dict['points2cc_dist_s']
#
#         pred_heatmap = output['pred_heatmap']
#         pred_loc = output['pred_loc']  # [B, 3, s_y, s_x]
#         pred_z = output['pred_z']
#         sample_idxs = output['sample_idxs']
#
#         # center_ind = ME.utils.sparse_quantize(coordinates=box_label[:, 0:2],
#         #                                       quantization_size=self.config.rpn_voxel_size[:2])  # [B, 2]
#         print(box_label[0])
#         center_ind = (box_label[:, 0:2] / torch.tensor(self.config.rpn_voxel_size[:2]).cuda()).floor()
#         b, _, s_y, s_x = pred_heatmap.size()
#
#         y, x = torch.meshgrid(torch.arange(s_y).cuda(), torch.arange(s_x).cuda())
#         x = x.unsqueeze(0).expand([b, s_y, s_x]) - s_x // 2
#         y = -(y.unsqueeze(0).expand([b, s_y, s_x]) - s_y // 2)  # consistent with the local coord sys.
#
#         # generate the ground-truth of heat map
#         x_ = x - center_ind[:, 0].view(b, 1, 1)
#         y_ = y - center_ind[:, 1].view(b, 1, 1)
#         dist_to_center = torch.abs(x_) + torch.abs(y_)  # Block metric
#         label_heatmap = torch.where(dist_to_center <= 2, torch.ones_like(y), torch.zeros_like(y))
#
#         # generate the ground-truth of regression
#         label_reg_x = box_label[:, 0:1].unsqueeze(-1) - x
#         label_reg_y = box_label[:, 1:2].unsqueeze(-1) - y
#         label_reg_r = box_label[:, 3:4].unsqueeze(-1).expand([b, s_y, s_x])
#         label_reg = torch.cat([label_reg_x.unsqueeze(1),
#                                label_reg_y.unsqueeze(1),
#                                label_reg_r.unsqueeze(1)],
#                               dim=1)  # [b, 3, w, h]
#
#         self.logger.experiment.add_images('heatmap_label', label_heatmap.unsqueeze(1)[0:4], self.global_step)
#         self.logger.experiment.add_images('heatmap_pred', pred_heatmap[0:4], self.global_step)
#         self.logger.experiment.add_mesh('pc_template', template[0:4], global_step=self.global_step)
#         self.logger.experiment.add_mesh('pc_search', search[0:4], global_step=self.global_step)
#
#         # generate the ground-truth of z-axis
#         label_z = box_label[:, 2:3].view(b, 1, 1, 1).expand([b, 1, s_y, s_x])
#
#         loss_heatmap = self.focal_loss(pred_heatmap.view(b, -1).unsqueeze(-1),
#                                        label_heatmap.view(b, -1).unsqueeze(-1),
#                                        torch.ones(b, s_x * s_y, 1).cuda())
#         loss_heatmap = loss_heatmap.mean()
#
#         loss_reg = F.smooth_l1_loss(pred_loc, label_reg, reduction='none')
#         loss_reg = (loss_reg * label_heatmap.unsqueeze(1)).sum() / (3 * label_heatmap.sum() + 1e-06)
#
#         loss_z = F.smooth_l1_loss(pred_z, label_z, reduction='none')
#         loss_z = (loss_z * label_heatmap.unsqueeze(1)).sum() / (label_heatmap.sum() + 1e-06)
#
#         out_dict = {
#             "loss_objective": loss_heatmap,
#             "loss_box": loss_reg,
#             "loss_seg": loss_z,
#             "loss_vote": loss_z,
#             "loss_bc": loss_z
#         }
#         return out_dict
#
#     def forward(self, input_dict):
#         """
#         :param input_dict:
#         {
#         'template_points': template_points.astype('float32'),
#         't_voxel_coords', 't_voxel_feats', 't_voxel_inds',
#         'search_points': search_points.astype('float32'),
#         's_voxel_coords', 's_voxel_feats', 's_voxel_inds',
#         'box_label': np.array(search_bbox_reg).astype('float32'),
#         'bbox_size': search_box.wlh,
#         'seg_label': seg_label.astype('float32'),
#         'points2cc_dist_t': template_bc,
#         'points2cc_dist_s': search_bc,
#         }
#
#         :return:
#         """
#         template = input_dict['template_points']
#         search = input_dict['search_points']
#         template_bc = input_dict['points2cc_dist_t']
#
#         t_voxel_coords, t_voxel_feats, t_voxel_inds \
#             = input_dict['t_voxel_coords'], input_dict['t_voxel_feats'], input_dict['t_voxel_inds']
#         s_voxel_coords, s_voxel_feats, s_voxel_inds \
#             = input_dict['s_voxel_coords'], input_dict['s_voxel_feats'], input_dict['s_voxel_inds']
#
#         # backbone, return # [B, N, 3], [B, C, N], [B, N]
#         template_xyz, template_feature, sample_idxs_t, _ \
#             = self.backbone(template, t_voxel_coords, t_voxel_feats, t_voxel_inds, self.num_sampled_point)
#
#         search_xyz, search_feature, sample_idxs, search_coords \
#             = self.backbone(search, s_voxel_coords, s_voxel_feats, s_voxel_inds, self.num_sampled_point)
#
#         # prepare bc
#         pred_search_bc = self.mlp_bc(torch.cat([search_xyz.transpose(1, 2), search_feature], dim=1))  # [B, 9, 128]
#         pred_search_bc = pred_search_bc.transpose(1, 2)
#         sample_idxs_t = sample_idxs_t[:, :, None]
#         template_bc = template_bc.gather(dim=1, index=sample_idxs_t.repeat(1, 1, template_bc.size(2)).long())
#
#         # point-voxel based xcorr
#         fusion_feature = self.xcorr(search_xyz, template_xyz,
#                                     search_feature, template_feature,
#                                     pred_search_bc, template_bc)
#
#         if self.config.use_bev:
#             # voxelize fusion_feature
#             fusion_feature = fusion_feature.transpose(1, 2).contiguous()  # B,N,C
#
#             # search_coords = torch.flatten(search_coords, 0, 1)
#             # fusion_feature = torch.flatten(fusion_feature, 0, 1)
#             batch_size = search_xyz.size(0)
#             fusion_coords_list, fusion_feat_list = [], []
#             rpn_voxel_size = torch.tensor(self.config.rpn_voxel_size, device=search.device)
#             for idx in range(batch_size):
#                 fusion_coords = (search_xyz[idx, :, :] / rpn_voxel_size).floor()
#                 _, fusion_idx = ME.utils.sparse_quantize(fusion_coords.cpu(), return_index=True)
#                 fusion_coords_list.append(fusion_coords[fusion_idx, :])
#                 fusion_feat_list.append(fusion_feature[idx, fusion_idx, :])
#             search_coords, fusion_feature = ME.utils.sparse_collate(fusion_coords_list, fusion_feat_list)
#
#             fusion_sparse = ME.SparseTensor(fusion_feature, search_coords, device=search.device)
#             fusion_sparse = self.fusion_trans(fusion_sparse)
#
#             # to dense
#             # min_coords = torch.tensor(self.config.min_bound) / torch.tensor(self.config.rpn_voxel_size)
#             # min_coords = torch.IntTensor(min_coords.floor().int())
#             # shape_dense = torch.Size([batch_size, -1, min_coords[0] * -2, min_coords[1] * -2, min_coords[2] * -2])
#             min_coords = torch.IntTensor(fusion_sparse.C[:, 1:].min(0)[0].cpu())
#             fusion_dense, _, _ = fusion_sparse.dense(min_coordinate=min_coords)  # [B, C, s_x, s_y, s_z]
#             fusion_dense = fusion_dense.permute(0, 1, 4, 3, 2).contiguous()  # out of memory????
#             fusion_dense_bev, _ = torch.max(fusion_dense, dim=2, keepdim=False)  # [B, C, s_y, s_x]
#
#             pred_heatmap, pred_loc, pred_theta = self.bev_rpn(fusion_dense_bev)
#
#             end_points = {'pred_heatmap': pred_heatmap,
#                           "pred_loc": pred_loc,
#                           'sample_idxs': sample_idxs,
#                           'pred_z': pred_theta,
#                           'pred_search_bc': pred_search_bc,
#                           }
#         else:
#             # proposal generation
#             estimation_boxes, estimation_cla, vote_xyz, center_xyzs, \
#             estimation_score, center_idxs = self.rpn(search_xyz, fusion_feature)
#
#             search_center_idxs = torch.gather(sample_idxs.long(), 1, center_idxs.long())
#
#             end_points = {"estimation_boxes": estimation_boxes,
#                           "center_xyz": center_xyzs,
#                           'sample_idxs': sample_idxs,
#                           'estimation_cla': estimation_cla,
#                           "vote_xyz": vote_xyz,
#                           "pred_search_bc": pred_search_bc,
#                           "estimation_score": estimation_score,
#                           "search_center_idxs": search_center_idxs,
#                           }
#
#         return end_points
#
#     def training_step(self, batch, batch_idx):
#
#         if self.global_step % 5 == 0:
#             torch.cuda.empty_cache()
#
#         end_points = self(batch)
#
#         # compute loss
#         if self.config.use_bev:
#             loss_dict = self.compute_loss_bev(batch, end_points)
#             loss = loss_dict['loss_objective'] * 1.0 \
#                    + loss_dict['loss_box'] * 1.0 \
#                    + loss_dict['loss_seg'] * 2.0 \
#                    + loss_dict['loss_vote'] * 0 \
#                    + loss_dict['loss_bc'] * 0
#         else:
#             loss_dict = self.compute_loss(batch, end_points)
#             loss = loss_dict['loss_objective'] * self.config.objectiveness_weight \
#                    + loss_dict['loss_box'] * self.config.box_weight \
#                    + loss_dict['loss_seg'] * self.config.seg_weight \
#                    + loss_dict['loss_vote'] * self.config.vote_weight \
#                    + loss_dict['loss_bc'] * self.config.bc_weight
#
#         # log
#         self.log('loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
#         self.log('loss_box/train', loss_dict['loss_box'].item(), on_step=True, on_epoch=True, prog_bar=True,
#                  logger=False)
#         self.log('loss_seg/train', loss_dict['loss_seg'].item(), on_step=True, on_epoch=True, prog_bar=True,
#                  logger=False)
#         self.log('loss_vote/train', loss_dict['loss_vote'].item(), on_step=True, on_epoch=True, prog_bar=True,
#                  logger=False)
#         self.log('loss_bc/train', loss_dict['loss_bc'].item(), on_step=True, on_epoch=True, prog_bar=True,
#                  logger=False)
#         self.log('loss_objective/train', loss_dict['loss_objective'].item(), on_step=True, on_epoch=True, prog_bar=True,
#                  logger=False)
#
#         self.logger.experiment.add_scalars('loss', {'loss_total': loss.item(),
#                                                     'loss_box': loss_dict['loss_box'].item(),
#                                                     'loss_seg': loss_dict['loss_seg'].item(),
#                                                     'loss_vote': loss_dict['loss_vote'].item(),
#                                                     'loss_objective': loss_dict['loss_objective'].item(),
#                                                     'loss_bc': loss_dict['loss_bc'].item()},
#                                            global_step=self.global_step)
#
#         return loss
#
#     def evaluate_one_sequence(self, sequence):
#         """
#
#         :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
#         :return:
#         """
#         ious = []
#         distances = []
#         results_bbs = []
#         model_state = None
#
#         for frame_id in range(len(sequence)):  # tracklet
#             this_bb = sequence[frame_id]["3d_bbox"]
#             if frame_id == 0:
#                 # the first frame
#                 results_bbs.append(this_bb)
#                 model_state = [0, 0, 0, this_bb.wlh[1], this_bb.wlh[0], this_bb.wlh[2], 0]
#                 model_state = torch.tensor(model_state, device=self.device, dtype=torch.float32)
#                 model_state = model_state.view(1, 1, 7)
#             else:
#
#                 # preparing search area
#                 search_pc_crop, ref_bb = self.generate_search_area(sequence, frame_id, results_bbs)
#
#                 # update template
#                 template_pc, canonical_box = self.generate_template(sequence, frame_id, results_bbs)
#
#                 # construct input dict
#                 data_dict = self.prepare_input(template_pc, search_pc_crop, canonical_box)
#
#                 end_points = self(data_dict)
#
#                 if self.config.use_bev:
#                     score = torch.sigmoid(end_points['pred_heatmap'])
#                     b, _, s_y, s_x = score.size()
#                     score = score.view(b, -1)  # [B, s_x * s_y]
#                     idx = torch.argmax(score, dim=1, keepdim=True)
#
#                     # gather z-axis value
#                     z = end_points['pred_z'].view(b, -1)
#                     z = torch.gather(z, dim=1, index=idx)
#
#                     # calculate and gather x-axis and y-axis values
#                     y, x = (idx // s_x), (idx % s_x)
#                     offset = end_points['pred_loc'].view(b, 3, -1)
#                     idx = idx.unsqueeze(1).repeat(1, 3, 1)
#                     best_offset = torch.gather(offset, dim=2, index=idx)  # [b, 3, 1]
#
#                     x = best_offset[:, 0, :] + x
#                     y = best_offset[:, 1, :] + y
#                     theta = best_offset[:, 2, :]
#
#                     selected_box = torch.cat([x, y, z, theta], dim=1).unsqueeze(1)  # [b, 1, 4]
#                 else:
#                     decode_box = self.box_coder.decode_torch_template_batch(end_points['estimation_boxes'],
#                                                                             end_points['center_xyz'],
#                                                                             model_state)  #[B, 64, 7]
#
#                     idx_box = torch.argmax(end_points['estimation_score'], dim=1, keepdim=True)  #
#                     selected_box = torch.gather(decode_box, 1,
#                                                 idx_box.repeat(1, 1, decode_box.shape[2]))  # [B, 1, 7]
#
#                 selected_box_cpu = selected_box.squeeze(1).detach().cpu().numpy()  # [B, 7]
#                 candidate_box = points_utils.getOffsetBB(ref_bb, selected_box_cpu[0],
#                                                          degrees=self.config.degrees,
#                                                          use_z=self.config.use_z,
#                                                          limit_box=self.config.limit_box)
#                 results_bbs.append(candidate_box)
#
#             this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
#                                            up_axis=self.config.up_axis)
#             this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
#                                              up_axis=self.config.up_axis)
#             ious.append(this_overlap)
#             distances.append(this_accuracy)
#         return ious, distances


# class PVT(base_model.BaseModel):
#     def __init__(self, config=None, **kwargs):
#         super().__init__(config, **kwargs)
#         self.save_hyperparameters()
#         self.num_sampled_point = self.config.num_sampled_points
#
#         if self.config.optimizer == 'sam':
#             self.automatic_optimization = False
#
#         self.backbone = SparseConvBackbone(input_feature_dim=self.config.input_channel,
#                                            output_feature_dim=self.config.feature_channel,
#                                            conv1_kernel_size=self.config.conv1_kernel_size,
#                                            checkpoint=self.config.checkpoint)
#
#         self.mlp_bc = (pt_utils.Seq(3 + self.config.feature_channel)
#                        .conv1d(self.config.feature_channel, bn=True)
#                        .conv1d(self.config.feature_channel, bn=True)
#                        .conv1d(self.config.bc_channel, activation=None))
#
#         # self.xcorr = BoxAwareXCorr(feature_channel=self.config.feature_channel,
#         #                            hidden_channel=self.config.hidden_channel,
#         #                            out_channel=self.config.out_channel,
#         #                            k=self.config.k,
#         #                            use_search_bc=self.config.use_search_bc,
#         #                            use_search_feature=self.config.use_search_feature,
#         #                            bc_channel=self.config.bc_channel)
#
#         self.xcorr = PointVoxelXCorr(num_levels=self.config.num_levels,
#                                      base_scale=self.config.base_scale,
#                                      truncate_k=self.config.truncate_k,
#                                      num_knn=self.config.num_knn,
#                                      feat_channel=self.config.out_channel // 2)
#
#         self.rpn = P2BVoteNetRPN(self.config.feature_channel,
#                                  vote_channel=self.config.vote_channel,
#                                  num_proposal=self.config.num_proposal,
#                                  normalize_xyz=self.config.normalize_xyz)
#
#
#     def prepare_input(self, template_pc, search_pc, template_box, *args, **kwargs):
#
#         template_points, idx_t = points_utils.regularize_pc(template_pc.points.T,
#                                                             self.config.template_size,
#                                                             seed=1)
#         search_points, idx_s = points_utils.regularize_pc(search_pc.points.T,
#                                                           self.config.search_size,
#                                                           seed=1)
#
#         # prepare sparse coordinates and features for search area
#         search_sparse_coords, search_sparse_feat, search_sparse_idx \
#             = ME.utils.sparse_quantize(search_points, search_points, return_index=True,
#                                        quantization_size=self.config.voxel_size)
#         if len(search_sparse_feat.shape) == 1:
#             search_sparse_feat = search_sparse_feat[np.newaxis, :]
#         search_sparse_coords, search_sparse_feat = ME.utils.sparse_collate([search_sparse_coords], [search_sparse_feat])
#
#         # prepare sparse coordinates and features for template points
#         tmpl_sparse_coords, tmpl_sparse_feat, tmpl_sparse_idx \
#             = ME.utils.sparse_quantize(template_points, template_points, return_index=True,
#                                        quantization_size=self.config.voxel_size)
#         if len(tmpl_sparse_feat.shape) == 1:
#             tmpl_sparse_feat = tmpl_sparse_feat[np.newaxis, :]
#
#         tmpl_sparse_coords, tmpl_sparse_feat = ME.utils.sparse_collate([tmpl_sparse_coords], [tmpl_sparse_feat])
#
#         template_points_torch = torch.tensor(template_points, device=self.device, dtype=torch.float32)
#         search_points_torch = torch.tensor(search_points, device=self.device, dtype=torch.float32)
#         template_bc = points_utils.get_point_to_box_distance(template_points, template_box)
#         template_bc_torch = torch.tensor(template_bc, device=self.device, dtype=torch.float32)
#
#         data_dict = {
#             'template_points': template_points_torch[None, ...],
#             't_voxel_coords': tmpl_sparse_coords,
#             't_voxel_feats': tmpl_sparse_feat,
#             't_voxel_inds': tmpl_sparse_idx,
#             'search_points': search_points_torch[None, ...],
#             's_voxel_coords': search_sparse_coords,
#             's_voxel_feats': search_sparse_feat,
#             's_voxel_inds': search_sparse_idx,
#             'points2cc_dist_t': template_bc_torch[None, ...]
#         }
#         return data_dict
#
#     def compute_loss(self, data, output):
#         search_bc = data['points2cc_dist_s']
#         estimation_cla = output['estimation_cla']  # B,N
#         N = estimation_cla.shape[1]
#         seg_label = data['seg_label']
#         sample_idxs = output['sample_idxs']  # B,N
#         seg_label = seg_label.gather(dim=1, index=sample_idxs[:, :N].long())  # B,N
#         search_bc = search_bc.gather(dim=1, index=sample_idxs[:, :N, None].repeat(1, 1, self.config.bc_channel).long())
#         # update label
#         data['seg_label'] = seg_label
#         data['points2cc_dist_s'] = search_bc
#
#         out_dict = super(PVT, self).compute_loss(data, output)
#         search_bc = data['points2cc_dist_s']
#         pred_search_bc = output['pred_search_bc']
#         seg_label = data['seg_label']
#         loss_bc = F.smooth_l1_loss(pred_search_bc, search_bc, reduction='none')
#         loss_bc = torch.sum(loss_bc.mean(2) * seg_label) / (seg_label.sum() + 1e-6)
#         out_dict["loss_bc"] = loss_bc
#         return out_dict
#
#     def forward(self, input_dict):
#         """
#         :param input_dict:
#         {
#         'template_points': template_points.astype('float32'),
#         'search_points': search_points.astype('float32'),
#         'box_label': np.array(search_bbox_reg).astype('float32'),
#         'bbox_size': search_box.wlh,
#         'seg_label': seg_label.astype('float32'),
#         'points2cc_dist_t': template_bc,
#         'points2cc_dist_s': search_bc,
#         }
#
#         :return:
#         """
#
#         template = input_dict['template_points']
#         search = input_dict['search_points']
#         template_bc = input_dict['points2cc_dist_t']
#
#         t_voxel_coords, t_voxel_feats, t_voxel_inds \
#             = input_dict['t_voxel_coords'], input_dict['t_voxel_feats'], input_dict['t_voxel_inds']
#         s_voxel_coords, s_voxel_feats, s_voxel_inds \
#             = input_dict['s_voxel_coords'], input_dict['s_voxel_feats'], input_dict['s_voxel_inds']
#
#         # backbone, return # [B, N, 3], [B, C, N], [B, N]
#         template_xyz, template_feature, sample_idxs_t, _ \
#             = self.backbone(template, t_voxel_coords,
#                             t_voxel_feats, t_voxel_inds,
#                             self.num_sampled_point // 2, self.config.use_feat_norm)
#         search_xyz, search_feature, sample_idxs, search_coords \
#             = self.backbone(search, s_voxel_coords,
#                             s_voxel_feats, s_voxel_inds,
#                             self.num_sampled_point, self.config.use_feat_norm)
#
#         # prepare bc
#         pred_search_bc = self.mlp_bc(torch.cat([search_xyz.transpose(1, 2), search_feature], dim=1))  # [B, 9, 128]
#         pred_search_bc = pred_search_bc.transpose(1, 2)
#         sample_idxs_t = sample_idxs_t[:, :, None]
#         template_bc = template_bc.gather(dim=1, index=sample_idxs_t.repeat(1, 1, template_bc.size(2)).long())
#
#         # box-aware xcorr
#         # fusion_feature = self.xcorr(template_feature, search_feature,
#         #                             template_xyz, search_xyz,
#         #                             template_bc, pred_search_bc)
#         fusion_feature = self.xcorr(search_xyz, template_xyz,
#                                     search_feature, template_feature,
#                                     pred_search_bc, template_bc)
#
#         # for decorrelation
#         fusion_feature_avg = F.avg_pool1d(fusion_feature, sample_idxs.size(1)).squeeze(2)  # B, C,
#
#         # proposal generation
#         estimation_boxes, estimation_cla, vote_xyz, center_xyzs = self.rpn(search_xyz, fusion_feature)
#
#         end_points = {"estimation_boxes": estimation_boxes,
#                       "vote_center": vote_xyz,
#                       "pred_seg_score": estimation_cla,
#                       "center_xyz": center_xyzs,
#                       'sample_idxs': sample_idxs,
#                       'estimation_cla': estimation_cla,
#                       "vote_xyz": vote_xyz,
#                       "pred_search_bc": pred_search_bc,
#                       "fusion_avg": fusion_feature_avg
#                       }
#         return end_points
#
#     def training_step(self, batch, batch_idx, dataloader_idx=None):
#         """
#         {"estimation_boxes": estimation_boxs.transpose(1, 2).contiguous(),
#                   "vote_center": vote_xyz,
#                   "pred_seg_score": estimation_cla,
#                   "center_xyz": center_xyzs,
#                   "seed_idxs":
#                   "seg_label"
#                   "pred_search_bc": pred_search_bc
#         }
#         """
#         if self.global_step % 100 == 0:
#             torch.cuda.empty_cache()
#
#         # compute loss
#         end_points = self(batch)
#         loss_dict = self.compute_loss(batch, end_points)
#         loss = loss_dict['loss_objective'] * self.config.objectiveness_weight \
#                + loss_dict['loss_box'] * self.config.box_weight \
#                + loss_dict['loss_seg'] * self.config.seg_weight \
#                + loss_dict['loss_vote'] * self.config.vote_weight \
#                + loss_dict['loss_bc'] * self.config.bc_weight
#
#         # log
#         self.log('loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
#         self.log('loss_box/train', loss_dict['loss_box'].item(), on_step=True, on_epoch=True, prog_bar=True,
#                  logger=False)
#         self.log('loss_seg/train', loss_dict['loss_seg'].item(), on_step=True, on_epoch=True, prog_bar=True,
#                  logger=False)
#         self.log('loss_vote/train', loss_dict['loss_vote'].item(), on_step=True, on_epoch=True, prog_bar=True,
#                  logger=False)
#         self.log('loss_bc/train', loss_dict['loss_bc'].item(), on_step=True, on_epoch=True, prog_bar=True,
#                  logger=False)
#         self.log('loss_objective/train', loss_dict['loss_objective'].item(), on_step=True, on_epoch=True, prog_bar=True,
#                  logger=False)
#
#         self.logger.experiment.add_scalars('loss', {'loss_total': loss.item(),
#                                                     'loss_box': loss_dict['loss_box'].item(),
#                                                     'loss_seg': loss_dict['loss_seg'].item(),
#                                                     'loss_vote': loss_dict['loss_vote'].item(),
#                                                     'loss_objective': loss_dict['loss_objective'].item(),
#                                                     'loss_bc': loss_dict['loss_bc'].item()},
#                                            global_step=self.global_step)
#
#         return loss


class PVT(base_model.BaseModel):
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

        self.xcorr = PointVoxelXCorr(num_levels=self.config.num_levels,
                                     base_scale=self.config.base_scale,
                                     num_knn=self.config.num_knn,
                                     feat_channel=self.config.out_channel)  # // 2

        # self.xcorr = P2B_XCorr(feature_channel=self.config.feature_channel,
        #                        hidden_channel=self.config.hidden_channel,
        #                        out_channel=self.config.out_channel)
        # self.xcorr = BoxAwareXCorr(feature_channel=self.config.feature_channel,
        #                            hidden_channel=self.config.hidden_channel,
        #                            out_channel=self.config.out_channel,
        #                            k=self.config.k,
        #                            use_search_bc=self.config.use_search_bc,
        #                            use_search_feature=self.config.use_search_feature,
        #                            bc_channel=self.config.bc_channel)

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

        fusion_feature = self.xcorr(search_xyz, template_xyz,
                                    search_feature, template_feature,
                                    # pred_search_bc, template_bc
                                    ) # pv
        # fusion_feature = self.xcorr(template_feature, search_feature, template_xyz)

        # fusion_feature = self.xcorr(template_feature, search_feature,
        #                             template_xyz, search_xyz,
        #                             template_bc, pred_search_bc) # bat

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
        # self.eval()
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

        # for saving testing boxes.
        save_test_path = getattr(self.config, 'save_test_path', None)
        if save_test_path is not None and self.config.test:
            save_test_path = os.path.join(self.config.save_test_path, self.config.category_name)
            os.makedirs(save_test_path, exist_ok=True)
            file_name = os.path.join(save_test_path, f'{self.tracklet_count:04.0f}.txt')
            results_bbs_saved = [b.corners().flatten() for b in results_bbs]
            np.savetxt(file_name, results_bbs_saved, fmt='%f')
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
