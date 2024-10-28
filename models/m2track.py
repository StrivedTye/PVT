"""
m2track.py
Created by zenn at 2021/11/24 13:10
"""
from datasets import points_utils
from models import base_model
from models.backbone.pointnet import MiniPointNet, SegPointNet

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os

from utils.metrics import estimateOverlap, estimateAccuracy
from torchmetrics import Accuracy
from nuscenes.utils import geometry_utils

from hausdorff import hausdorff_distance
from chamferdist import ChamferDistance


class M2TRACK(base_model.BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()
        self.seg_acc = Accuracy(num_classes=2, average='none')

        self.box_aware = getattr(config, 'box_aware', False)
        self.use_motion_cls = getattr(config, 'use_motion_cls', True)
        self.use_second_stage = getattr(config, 'use_second_stage', True)
        self.use_prev_refinement = getattr(config, 'use_prev_refinement', True)
        self.seg_pointnet = SegPointNet(input_channel=3 + 1 + 1 + (9 if self.box_aware else 0),
                                        per_point_mlp1=[64, 64, 64, 128, 1024],
                                        per_point_mlp2=[512, 256, 128, 128],
                                        output_size=2 + (9 if self.box_aware else 0))
        self.mini_pointnet = MiniPointNet(input_channel=3 + 1 + (9 if self.box_aware else 0),
                                          per_point_mlp=[64, 128, 256, 512],
                                          hidden_mlp=[512, 256],
                                          output_size=-1)
        if self.use_second_stage:
            self.mini_pointnet2 = MiniPointNet(input_channel=3 + (9 if self.box_aware else 0),
                                               per_point_mlp=[64, 128, 256, 512],
                                               hidden_mlp=[512, 256],
                                               output_size=-1)

            self.box_mlp = nn.Sequential(nn.Linear(256, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 4))
        if self.use_prev_refinement:
            self.final_mlp = nn.Sequential(nn.Linear(256, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 4))
        if self.use_motion_cls:
            self.motion_state_mlp = nn.Sequential(nn.Linear(256, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 2))
            self.motion_acc = Accuracy(num_classes=2, average='none')

        self.motion_mlp = nn.Sequential(nn.Linear(256, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 4))

    def forward(self, input_dict):
        """
        Args:
            input_dict: {
            "points": (B,N,3+1+1)
            "candidate_bc": (B,N,9)

        }

        Returns: B,4

        """
        output_dict = {}
        x = input_dict["points"].transpose(1, 2)
        if self.box_aware:
            candidate_bc = input_dict["candidate_bc"].transpose(1, 2)
            x = torch.cat([x, candidate_bc], dim=1)

        B, _, N = x.shape

        seg_out = self.seg_pointnet(x)
        seg_logits = seg_out[:, :2, :]  # B,2,N
        pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True)  # B,1,N
        mask_points = x[:, :4, :] * pred_cls
        mask_xyz_t0 = mask_points[:, :3, :N // 2]  # B,3,N//2
        mask_xyz_t1 = mask_points[:, :3, N // 2:]
        if self.box_aware:
            pred_bc = seg_out[:, 2:, :]
            mask_pred_bc = pred_bc * pred_cls
            # mask_pred_bc_t0 = mask_pred_bc[:, :, :N // 2]  # B,9,N//2
            # mask_pred_bc_t1 = mask_pred_bc[:, :, N // 2:]
            mask_points = torch.cat([mask_points, mask_pred_bc], dim=1)
            output_dict['pred_bc'] = pred_bc.transpose(1, 2)

        point_feature = self.mini_pointnet(mask_points)

        # motion state prediction
        motion_pred = self.motion_mlp(point_feature)  # B,4
        if self.use_motion_cls:
            motion_state_logits = self.motion_state_mlp(point_feature)  # B,2
            motion_mask = torch.argmax(motion_state_logits, dim=1, keepdim=True)  # B,1
            motion_pred_masked = motion_pred * motion_mask
            output_dict['motion_cls'] = motion_state_logits
        else:
            motion_pred_masked = motion_pred
        # previous bbox refinement
        if self.use_prev_refinement:
            prev_boxes = self.final_mlp(point_feature)  # previous bb, B,4
            output_dict["estimation_boxes_prev"] = prev_boxes[:, :4]
        else:
            prev_boxes = torch.zeros_like(motion_pred)

        # 1st stage prediction
        aux_box = points_utils.get_offset_box_tensor(prev_boxes, motion_pred_masked)

        # 2nd stage refinement
        if self.use_second_stage:
            mask_xyz_t0_2_t1 = points_utils.get_offset_points_tensor(mask_xyz_t0.transpose(1, 2),
                                                                     prev_boxes[:, :4],
                                                                     motion_pred_masked).transpose(1, 2)  # B,3,N//2
            mask_xyz_t01 = torch.cat([mask_xyz_t0_2_t1, mask_xyz_t1], dim=-1)  # B,3,N

            # transform to the aux_box coordinate system
            mask_xyz_t01 = points_utils.remove_transform_points_tensor(mask_xyz_t01.transpose(1, 2),
                                                                       aux_box).transpose(1, 2)

            if self.box_aware:
                mask_xyz_t01 = torch.cat([mask_xyz_t01, mask_pred_bc], dim=1)
            output_offset = self.box_mlp(self.mini_pointnet2(mask_xyz_t01))  # B,4
            output = points_utils.get_offset_box_tensor(aux_box, output_offset)
            output_dict["estimation_boxes"] = output
        else:
            output_dict["estimation_boxes"] = aux_box

        output_dict.update({"seg_logits": seg_logits,
                            "motion_pred": motion_pred,
                            'aux_estimation_boxes': aux_box,
                            })

        return output_dict

    def compute_loss(self, data, output):
        loss_total = 0.0
        loss_dict = {}
        aux_estimation_boxes = output['aux_estimation_boxes']  # B,4
        motion_pred = output['motion_pred']  # B,4
        seg_logits = output['seg_logits']
        with torch.no_grad():
            seg_label = data['seg_label']
            box_label = data['box_label']
            box_label_prev = data['box_label_prev']
            motion_label = data['motion_label']
            motion_state_label = data['motion_state_label']
            center_label = box_label[:, :3]
            angle_label = torch.sin(box_label[:, 3])
            center_label_prev = box_label_prev[:, :3]
            angle_label_prev = torch.sin(box_label_prev[:, 3])
            center_label_motion = motion_label[:, :3]
            angle_label_motion = torch.sin(motion_label[:, 3])

        loss_seg = F.cross_entropy(seg_logits, seg_label, weight=torch.tensor([0.5, 2.0]).cuda())
        if self.use_motion_cls:
            motion_cls = output['motion_cls']  # B,2
            loss_motion_cls = F.cross_entropy(motion_cls, motion_state_label)
            loss_total += loss_motion_cls * self.config.motion_cls_seg_weight
            loss_dict['loss_motion_cls'] = loss_motion_cls

            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion, reduction='none')
            loss_center_motion = (motion_state_label * loss_center_motion.mean(dim=1)).sum() / (
                    motion_state_label.sum() + 1e-6)
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion, reduction='none')
            loss_angle_motion = (motion_state_label * loss_angle_motion).sum() / (motion_state_label.sum() + 1e-6)
        else:
            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion)
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion)

        if self.use_second_stage:
            estimation_boxes = output['estimation_boxes']  # B,4
            loss_center = F.smooth_l1_loss(estimation_boxes[:, :3], center_label)
            loss_angle = F.smooth_l1_loss(torch.sin(estimation_boxes[:, 3]), angle_label)
            loss_total += 1 * (loss_center * self.config.center_weight + loss_angle * self.config.angle_weight)
            loss_dict["loss_center"] = loss_center
            loss_dict["loss_angle"] = loss_angle
        if self.use_prev_refinement:
            estimation_boxes_prev = output['estimation_boxes_prev']  # B,4
            loss_center_prev = F.smooth_l1_loss(estimation_boxes_prev[:, :3], center_label_prev)
            loss_angle_prev = F.smooth_l1_loss(torch.sin(estimation_boxes_prev[:, 3]), angle_label_prev)
            loss_total += (loss_center_prev * self.config.center_weight + loss_angle_prev * self.config.angle_weight)
            loss_dict["loss_center_prev"] = loss_center_prev
            loss_dict["loss_angle_prev"] = loss_angle_prev

        loss_center_aux = F.smooth_l1_loss(aux_estimation_boxes[:, :3], center_label)

        loss_angle_aux = F.smooth_l1_loss(torch.sin(aux_estimation_boxes[:, 3]), angle_label)

        loss_total += loss_seg * self.config.seg_weight \
                      + 1 * (loss_center_aux * self.config.center_weight + loss_angle_aux * self.config.angle_weight) \
                      + 1 * (
                              loss_center_motion * self.config.center_weight + loss_angle_motion * self.config.angle_weight)
        loss_dict.update({
            "loss_total": loss_total,
            "loss_seg": loss_seg,
            "loss_center_aux": loss_center_aux,
            "loss_center_motion": loss_center_motion,
            "loss_angle_aux": loss_angle_aux,
            "loss_angle_motion": loss_angle_motion,
        })
        if self.box_aware:
            prev_bc = data['prev_bc']
            this_bc = data['this_bc']
            bc_label = torch.cat([prev_bc, this_bc], dim=1)
            pred_bc = output['pred_bc']
            loss_bc = F.smooth_l1_loss(pred_bc, bc_label)
            loss_total += loss_bc * self.config.bc_weight
            loss_dict.update({
                "loss_total": loss_total,
                "loss_bc": loss_bc
            })

        return loss_dict

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: {
            "points": stack_frames, (B,N,3+9+1)
            "seg_label": stack_label,
            "box_label": np.append(this_gt_bb_transform.center, theta),
            "box_size": this_gt_bb_transform.wlh
        }
        Returns:

        """
        output = self(batch)
        loss_dict = self.compute_loss(batch, output)
        loss = loss_dict['loss_total']

        # log
        seg_acc = self.seg_acc(torch.argmax(output['seg_logits'], dim=1, keepdim=False), batch['seg_label'])
        self.log('seg_acc_background/train', seg_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('seg_acc_foreground/train', seg_acc[1], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if self.use_motion_cls:
            motion_acc = self.motion_acc(torch.argmax(output['motion_cls'], dim=1, keepdim=False),
                                         batch['motion_state_label'])
            self.log('motion_acc_static/train', motion_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('motion_acc_dynamic/train', motion_acc[1], on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)

        log_dict = {k: v.item() for k, v in loss_dict.items()}

        self.logger.experiment.add_scalars('loss', log_dict,
                                           global_step=self.global_step)
        return loss

    def evaluate_one_sequence(self, sequence):
        """
        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []
        results_bbs = []
        cd_imperception = []
        hd_imperception = []

        for frame_id in range(len(sequence)):  # tracklet
            this_bb = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
            else:

                # construct input dict
                data_dict, ref_bb = self.build_input_dict(sequence, frame_id, results_bbs)

                # attack
                if getattr(self.config, 'attack', False):
                    adv_path = './adv_examples/m2track/' + 'ba3_' + self.config.category_name + '/'
                    os.makedirs(adv_path, exist_ok=True)
                    search_points_clean = data_dict['points'][0, self.config.point_sample_size:, :3].clone()
                    np.savetxt(adv_path+f'clean_{self.tracklet_count:04.0f}_{frame_id:04d}.txt', search_points_clean.detach().cpu().numpy(), fmt='%f')

                    # data_dict = self.attack_pgd(data_dict, this_bb, results_bbs[-1])
                    # data_dict = self.attack_fgsm(data_dict, this_bb, results_bbs[-1])
                    # data_dict = self.attack_gauss(data_dict)
                    # data_dict = self.attack_cw(data_dict, this_bb, results_bbs[-1])

                    # data_dict = self.attack_trans(data_dict)
                    data_dict = self.attack_ba(data_dict, this_bb, results_bbs[-1])
                    # data_dict = self.attacker.attack_fgsm(data_dict, this_bb, results_bbs[-1], victim="m2track")
                    # data_dict = self.attacker.attack_pgd(data_dict, this_bb, results_bbs[-1], victim="m2track")
                    # data_dict = self.attacker.attack_cw(data_dict, this_bb, results_bbs[-1], victim="m2track")


                    search_points_adv = data_dict['points'][0, self.config.point_sample_size:, :3]
                    np.savetxt(adv_path+f'adv_{self.tracklet_count:04.0f}_{frame_id:04d}.txt', search_points_adv.detach().cpu().numpy(), fmt='%f')

                    hd_imperception.append(hausdorff_distance(search_points_adv.detach().cpu().numpy(),
                                                              search_points_clean.detach().cpu().numpy(),
                                                              distance='euclidean'))
                    cd_imperception.append(self.chamferdist(search_points_clean.unsqueeze(0),
                                                            search_points_adv.unsqueeze(0),
                                                            bidirectional=True).detach().cpu().item())

                # run the tracker
                end_points = self(data_dict)

                estimation_box = end_points['estimation_boxes']
                estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
                if len(estimation_box.shape) == 3:
                    best_box_idx = estimation_box_cpu[:, 4].argmax()
                    estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]

                candidate_box = points_utils.getOffsetBB(ref_bb, estimation_box_cpu, degrees=self.config.degrees,
                                                         use_z=self.config.use_z,
                                                         limit_box=self.config.limit_box)

                results_bbs.append(candidate_box)

            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)
        return ious, distances, hd_imperception, cd_imperception

    def build_input_dict(self, sequence, frame_id, results_bbs):
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

        prev_frame = sequence[frame_id - 1]
        this_frame = sequence[frame_id]
        prev_pc = prev_frame['pc']
        this_pc = this_frame['pc']
        ref_box = results_bbs[-1]
        prev_frame_pc = points_utils.generate_subwindow(prev_pc, ref_box,
                                                        scale=self.config.bb_scale,
                                                        offset=self.config.bb_offset)
        this_frame_pc = points_utils.generate_subwindow(this_pc, ref_box,
                                                        scale=self.config.bb_scale,
                                                        offset=self.config.bb_offset)

        canonical_box = points_utils.transform_box(ref_box, ref_box)
        prev_points, idx_prev = points_utils.regularize_pc(prev_frame_pc.points.T,
                                                           self.config.point_sample_size,
                                                           seed=1)

        this_points, idx_this = points_utils.regularize_pc(this_frame_pc.points.T,
                                                           self.config.point_sample_size,
                                                           seed=1)
        seg_mask_prev = geometry_utils.points_in_box(canonical_box, prev_points.T, 1.25).astype(float)

        # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
        # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
        if frame_id != 1:
            seg_mask_prev[seg_mask_prev == 0] = 0.2
            seg_mask_prev[seg_mask_prev == 1] = 0.8
        seg_mask_this = np.full(seg_mask_prev.shape, fill_value=0.5)

        timestamp_prev = np.full((self.config.point_sample_size, 1), fill_value=0)
        timestamp_this = np.full((self.config.point_sample_size, 1), fill_value=0.1)
        prev_points = np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]], axis=-1)
        this_points = np.concatenate([this_points, timestamp_this, seg_mask_this[:, None]], axis=-1)

        stack_points = np.concatenate([prev_points, this_points], axis=0)

        data_dict = {"points": torch.tensor(stack_points[None, :], device=self.device, dtype=torch.float32),
                     }
        if getattr(self.config, 'box_aware', False):
            candidate_bc_prev = points_utils.get_point_to_box_distance(
                stack_points[:self.config.point_sample_size, :3], canonical_box)
            candidate_bc_this = np.zeros_like(candidate_bc_prev)
            candidate_bc = np.concatenate([candidate_bc_prev, candidate_bc_this], axis=0)
            data_dict.update({'candidate_bc': points_utils.np_to_torch_tensor(candidate_bc.astype('float32'),
                                                                              device=self.device)})
        return data_dict, results_bbs[-1]

    def run_track(self, data_dict, ref_bb):
        # run the tracker
        end_points = self(data_dict)

        estimation_box = end_points['estimation_boxes']
        estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
        if len(estimation_box.shape) == 3:
            best_box_idx = estimation_box_cpu[:, 4].argmax()
            estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]

        candidate_box = points_utils.getOffsetBB(ref_bb, estimation_box_cpu,
                                                 degrees=self.config.degrees,
                                                 use_z=self.config.use_z,
                                                 limit_box=self.config.limit_box)
        return candidate_box, end_points

    def attack_loss(self, end_point, gt_box, prev_box, target_attack=True):
        gt_box_normal = points_utils.transform_box(gt_box, prev_box)
        pred_boxes = end_point['estimation_boxes']
        center, wlh = gt_box_normal.center, gt_box_normal.wlh

        if target_attack:
            adv_center = torch.tensor(center + wlh, dtype=torch.float32).cuda()
            loss = F.smooth_l1_loss(pred_boxes[:, :3], adv_center[None, :])
            return -loss # for gradien ascend
        else:
            center = torch.tensor(center, dtype=torch.float32).cuda()
            loss = F.smooth_l1_loss(pred_boxes[:, :3], center[None, :])
            return loss

    def attack_fgsm(self, data_dict, gt_box, prev_box):
        epsilon = torch.tensor([0.05, 0.05, 0.05, 0, 0]).view(1, 1, -1).to(data_dict['points'])
        with torch.enable_grad():
            data_dict['points'] = data_dict['points'].clone().detach().requires_grad_(True)
            candidate_box, end_points = self.run_track(data_dict, prev_box)
            loss = self.attack_loss(end_points, gt_box, prev_box, target_attack=True)
            self.zero_grad()
            loss.backward()
            pertubation = epsilon * torch.sign(data_dict['points'].grad)
            data_dict['points'] = data_dict['points'] + pertubation
        return data_dict

    def attack_pgd(self, data_dict, gt_box, prev_box):
        epsilon = torch.tensor([0.05, 0.05, 0.05, 0, 0]).view(1, 1, -1).to(data_dict['points'])
        pgd_iter = 10
        with torch.enable_grad():
            data_dict['points'] = data_dict['points'] + (torch.rand_like(data_dict['points']) * 0.1 - 0.05)
            for i in range(pgd_iter):
                data_dict['points'] = data_dict['points'].clone().detach().requires_grad_(True)
                candidate_box, end_points = self.run_track(data_dict, prev_box)
                loss = self.attack_loss(end_points, gt_box, prev_box, target_attack=True)
                self.zero_grad()
                loss.backward()
                pertubation = epsilon * torch.sign(data_dict['points'].grad)
                data_dict['points'] = data_dict['points'] + pertubation
        return data_dict

    def attack_gauss(self, data_dict):
        epsilon = 0.05
        pertubation = epsilon * torch.randn_like(data_dict['points']).sign()
        data_dict['points'][:, :, :3] = data_dict['points'][:, :, :3] + pertubation[:, :, :3]
        return data_dict

    def attack_impluse(self, data_dict):
        p = 0.05 # probability
        N = data_dict['points'].size(1)
        rnd = torch.rand(N).cuda()
        mask_low = rnd < p
        mask_high = rnd > (1-p)

        data_dict['points'][:, mask_low, :3] = data_dict['points'][:, mask_low, :3] * 0
        data_dict['points'][:, mask_high, :3] = data_dict['points'][:, mask_high, :3] * 2
        return data_dict

    def attack_cw(self, data_dict, gt_box, prev_box):
        cw_iter = 10
        c = 1
        lr = 0.01

        pc = data_dict['points'][:, :, :3].clone().detach()
        w = 0.05 * torch.ones_like(pc).requires_grad_(True)
        w = torch.nn.Parameter(w)
        prev_cost = 1e6

        optimizer = torch.optim.Adam([w], lr=lr)

        with torch.enable_grad():
            for i in range(cw_iter):

                # adversarial example
                adv_pc = pc + w

                # calculate loss
                current_L2 = F.mse_loss(adv_pc, pc, reduction='none')
                L2_loss = current_L2.sum(dim=(1, 2)).mean()

                data_dict['points'] = torch.cat([adv_pc, data_dict['points'][:, :, 3:]], dim=2)
                candidate_box, end_points = self.run_track(data_dict, prev_box)

                # f
                f = -self.attack_loss(end_points, gt_box, prev_box, target_attack=True)

                cost = L2_loss + c * f

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                # Early stop when loss does not converge.
                if i % 3 == 0:
                    if cost.item() > prev_cost:
                        break
                    prev_cost = cost.item()
        data_dict['points'][:, :, :3] = pc + w
        return data_dict

    def attack_ba(self, data_dict, gt_box, prev_box):
        iter_num = 10
        c = 0.5
        lr = 0.01

        data_dict['search_points'] = data_dict['points'][:, self.config.point_sample_size:, :3]
        data_dict['template_points'] = data_dict['points'][:, :self.config.point_sample_size, :3]
        data_dict['points2cc_dist_t'] = data_dict['candidate_bc'][:, :self.config.point_sample_size, :]

        pc = data_dict['search_points'].clone().detach()
        w = 0.1 * torch.rand_like(pc).requires_grad_(True) - 0.05
        w = torch.nn.Parameter(w)

        prev_cost = 1e6

        optimizer = torch.optim.Adam([w], lr=lr)

        # object-aware
        # end_points = self.attacker.baseline.forward(data_dict)
        # candidate_box, end_points = self.attacker.run_track(end_points, prev_box)
        # canonical_box = points_utils.transform_box(candidate_box, prev_box)
        # important_mask = points_utils.get_in_box_mask(pc.cpu().numpy()[0, :, :].T, canonical_box)
        # important_mask = torch.from_numpy(important_mask)
        # pc[:, important_mask, 2] = pc[:, important_mask, 2] + 0.1

        with torch.enable_grad():
            for i in range(iter_num):

                # adversarial example
                adv_pc = pc + w

                # calculate loss
                cd_loss = 0.5 * self.chamferdist(adv_pc, pc, bidirectional=True)

                data_dict['search_points'] = adv_pc
                # _, end_points_ori = self.run_track(data_dict, prev_box)
                # f0 = self.attack_loss(end_points_ori, gt_box, prev_box, target_attack=True)

                end_points = self.attacker.baseline.forward(data_dict)
                candidate_box, end_points = self.attacker.run_track(end_points, prev_box)
                f_1 = -self.attacker.attack_loss(end_points, gt_box, prev_box, target_attack=True)

                mask_one_w, mask_zero_w = self.sample(w, 0.5)

                data_dict['search_points'] = pc + mask_one_w
                # candidate_box, end_points = self.run_track(data_dict, prev_box)
                end_points = self.attacker.baseline.forward(data_dict)
                candidate_box, end_points = self.attacker.run_track(end_points, prev_box)
                f_2 = -self.attacker.attack_loss(end_points, gt_box, prev_box, target_attack=True)

                data_dict['search_points'] = pc + mask_zero_w
                # candidate_box, end_points = self.run_track(data_dict, prev_box)
                end_points = self.attacker.baseline.forward(data_dict)
                candidate_box, end_points = self.attacker.run_track(end_points, prev_box)
                f_3 = -self.attacker.attack_loss(end_points, gt_box, prev_box, target_attack=True)

                # cost = cd_loss + c * (f_1 + f_2 + f_3)
                cost = cd_loss + c * f_1

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                # Early stop when loss does not converge.
                if i % 3 == 0:
                    if cost.item() > prev_cost:
                        break
                    prev_cost = cost.item()
            adv_pc = pc + w
            data_dict['points'][:, :, :3] = torch.cat([data_dict['template_points'], adv_pc], dim=1)
        return data_dict

    def attack_trans(self, data_dict):
        search_points = data_dict['points'][:, self.config.point_sample_size:, :3]
        template_points = data_dict['points'][:, :self.config.point_sample_size, :3]
        search_points_adv = self.attacker.forward_test(search_points, template_points)

        data_dict['points'][:, :, :3] = torch.cat([template_points, search_points_adv], dim=1)

        return data_dict







