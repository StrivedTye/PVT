""" 
baseModel.py
Created by zenn at 2021/5/9 14:40
"""
import torch
from easydict import EasyDict
import pytorch_lightning as pl
from datasets import points_utils
from utils.metrics import TorchSuccess, TorchPrecision, TorchAverage
from utils.metrics import estimateOverlap, estimateAccuracy
import torch.nn.functional as F

from utils.reweighting import weight_learner
from utils.sam import SAM

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import time

import yaml
from hausdorff import hausdorff_distance
from chamferdist import ChamferDistance


def vis(PC, PC_seeds, PC_seeds_vote, boxs, gt_box, gt_seed_cls, fg_score, fusion_feat=None):
    boxs = [boxs]
    ColorNames = ['#4682b4',  # stealblue
                  '#FF8C00',  # darkorange
                  '#808000',  # olive
                  'black',
                  '#8FBC8F',  # darkseagreen
                  'red',
                  ]

    # Create figure for TRACKING
    fig = plt.figure(figsize=(9, 6), facecolor="white")
    plt.rcParams['savefig.dpi'] = 320
    plt.rcParams['figure.dpi'] = 320

    # Create axis in 3D
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot the cropped point cloud
    ratio = 1
    sv = ax.scatter(
        PC[::ratio, 0],
        PC[::ratio, 1],
        PC[::ratio, 2],
        s=10,
        color="gray")

    # sv = ax.scatter(
    #     PC_seeds[:, 0],
    #     PC_seeds[:, 1],
    #     PC_seeds[:, 2],
    #     s=15,
    #     c=fg_score[0, :, 0])

    # sv = ax.scatter(
    #     PC_seeds_vote[:, 0],
    #     PC_seeds_vote[:, 1],
    #     PC_seeds_vote[:, 2],
    #     s=5,
    #     color='cyan')

    # sv = ax.scatter(
    #     PC_seeds[:, 0],
    #     PC_seeds[:, 1],
    #     PC_seeds[:, 2],
    #     s=15,
    #     c=fusion_feat[0, :])

    # plot the arrow that is from original seed to voted seed
    # for i, j, score in zip(PC_seeds, PC_seeds_vote, fg_score[0, :,  0]):
    #     if score < max(fg_score[0, :,  0])/2: continue
    #
    #     ax.plot([i[0], j[0]], [i[1], j[1]], [i[2], j[2]],
    #         color='gray',
    #         alpha=0.5,
    #         linewidth=1.3,
    #         marker='.',
    #         linestyle="-")

    # plot the foreground seed points
    # PC_pos_seeds = PC_seeds_vote[gt_seed_cls==1]
    # ax.scatter(
    #     PC_pos_seeds[:, 0],
    #     PC_pos_seeds[:, 1],
    #     PC_pos_seeds[:, 2],
    #     s=50,
    #     color="black")
    #
    # PC_pos_seeds = PC_seeds[gt_seed_cls==1]
    # ax.scatter(
    #     PC_pos_seeds[:, 0],
    #     PC_pos_seeds[:, 1],
    #     PC_pos_seeds[:, 2],
    #     s=50,
    #     color="red")

    # # point order to draw a full Box
    order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]
    #
    # # Plot GT box
    # ax.plot(
    #     gt_box.corners()[0, order],
    #     gt_box.corners()[1, order],
    #     gt_box.corners()[2, order],
    #     color='red',
    #     alpha=0.5,
    #     linewidth=5,
    #     linestyle="-")
    #
    # for id, box in enumerate(boxs):
    #     box = box.corners()
    #     ax.plot(box[0, order], box[1, order], box[2, order],
    #             color=ColorNames[id%6],
    #             alpha=1,
    #             linewidth=1,
    #             linestyle="-")

    ax.view_init(90, 0)
    # ax.view_init(30, -30)
    plt.axis('off')
    plt.colorbar(sv, ax=ax)
    # plt.ion()
    plt.show()
    # plt.pause(0.001)
    # plt.clf()

    # fig.canvas.draw()
    # res = np.array(fig.canvas.renderer._renderer).transpose([2, 0, 1])
    # plt.close()
    return fig


class BaseModel(pl.LightningModule):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = EasyDict(kwargs)
        self.config = config

        # testing metrics
        self.prec = TorchPrecision()
        self.success = TorchSuccess()
        self.hd = TorchAverage()
        self.cd = TorchAverage()

        self.times_tracklet = []
        self.tracklet_count = 0

        if getattr(self.config, "re_weight", False):
            # self.pre_features = torch.zeros(self.config.batch_size, self.config.feature_channel)
            # self.pre_weight = torch.ones(self.config.batch_size, 1)
            self.register_buffer('pre_features', torch.zeros(self.config.batch_size, self.config.feature_channel), persistent=False)
            self.register_buffer('pre_weight', torch.ones(self.config.batch_size, 1), persistent=False)

        if getattr(self.config, 'attack', False):
            from models.attack3d import ATTACK3D
            with open('cfgs/ATTACK.yaml', 'r') as f:
                try:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                except:
                    config = yaml.load(f)
            config = EasyDict(config)

            # self.attacker = ATTACK3D.load_from_checkpoint(config.attack_ckpt, config=config)
            self.attacker = ATTACK3D(config)
            self.attacker.eval()
            self.chamferdist = ChamferDistance()

    def configure_optimizers(self):
        if self.config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd)
        elif self.config.optimizer.lower() == 'sam':
            optimizer = SAM(self.parameters(), torch.optim.SGD, lr=self.config.lr, momentum=0.9)
        else:
            betas = getattr(self.config, 'betas', (0.5, 0.999))
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=tuple(betas), eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def compute_loss(self, data, output):
        """

        :param data: input data
        :param output:
        :return:
        """
        estimation_boxes = output['estimation_boxes']  # B,num_proposal,5
        estimation_cla = output['estimation_cla']  # B,N
        proposal_center = output["center_xyz"]  # B,num_proposal,3
        vote_xyz = output["vote_xyz"]
        seg_label = data['seg_label']
        box_label = data['box_label']  # B,4

        loss_seg = F.binary_cross_entropy_with_logits(estimation_cla, seg_label, reduction='none')  # B, N

        loss_vote = F.smooth_l1_loss(vote_xyz, box_label[:, None, :3].expand_as(vote_xyz), reduction='none')  # B,N,3
        # loss_vote = (loss_vote.mean(2) * seg_label).sum() / (seg_label.sum() + 1e-06)

        dist = torch.sum((proposal_center - box_label[:, None, :3]) ** 2, dim=-1)

        dist = torch.sqrt(dist + 1e-6)  # B, K
        objectness_label = torch.zeros_like(dist, dtype=torch.float32)
        objectness_label[dist < 0.3] = 1

        objectness_score = estimation_boxes[:, :, 4]  # B, K
        objectness_mask = torch.zeros_like(objectness_label, dtype=torch.float32)
        objectness_mask[dist < 0.3] = 1
        objectness_mask[dist > 0.6] = 1

        loss_objective = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
                                                            pos_weight=torch.tensor([2.0], device=self.device))  #B, K
        # loss_objective = torch.sum(loss_objective * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)

        loss_box = F.smooth_l1_loss(estimation_boxes[:, :, :4],
                                    box_label[:, None, :4].expand_as(estimation_boxes[:, :, :4]),
                                    reduction='none')  # B, K, 4
        # loss_box = torch.sum(loss_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)

        if getattr(self.config, "re_weight", False):
            cur_features = output["fusion_avg"]
            pre_features, pre_weight = self.pre_features, self.pre_weight

            if self.current_epoch >= self.config.rw_warmup:
                weight, pre_features, pre_weight = weight_learner(cur_features, pre_features, pre_weight,
                                                                  self.config, self.current_epoch, self.global_step)
            else:
                weight = 1 / cur_features.size(0) * torch.ones(cur_features.size(0), 1).cuda()
            self.pre_features.data.copy_(pre_features)
            self.pre_weight.data.copy_(pre_weight)

            loss_seg = loss_seg.mean(1).view(1, -1).mm(weight)

            loss_vote = (loss_vote.mean(2) * seg_label).sum(1) / (seg_label.sum(1) + 1e-6)
            loss_vote = loss_vote.view(1, -1).mm(weight)

            loss_objective = (loss_objective * objectness_mask).sum(1) / (objectness_mask.sum(1) + 1e-6)
            loss_objective = loss_objective.view(1, -1).mm(weight)

            loss_box = (loss_box.mean(2) * objectness_label).sum(1) / (objectness_label.sum(1) + 1e-6)
            loss_box = loss_box.view(1, -1).mm(weight)
        else:
            loss_seg = loss_seg.mean()
            loss_vote = (loss_vote.mean(2) * seg_label).sum() / (seg_label.sum() + 1e-06)
            loss_objective = torch.sum(loss_objective * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
            loss_box = torch.sum(loss_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)

        loss_dict = {
            "loss_objective": loss_objective,
            "loss_box": loss_box,
            "loss_seg": loss_seg,
            "loss_vote": loss_vote,
        }

        return loss_dict

    def generate_template(self, sequence, current_frame_id, results_bbs):
        """
        generate template for evaluating.
        the template can be updated using the previous predictions.
        :param sequence: the list of the whole sequence
        :param current_frame_id:
        :param results_bbs: predicted box for previous frames
        :return:
        """
        first_pc = sequence[0]['pc']
        previous_pc = sequence[current_frame_id - 1]['pc']
        if "firstandprevious".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.getModel([first_pc, previous_pc],
                                                               [results_bbs[0], results_bbs[current_frame_id - 1]],
                                                               scale=self.config.model_bb_scale,
                                                               offset=self.config.model_bb_offset)
        elif "first".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.cropAndCenterPC(first_pc, results_bbs[0],
                                                                      scale=self.config.model_bb_scale,
                                                                      offset=self.config.model_bb_offset)
        elif "previous".upper() in self.config.hape_aggregation.upper():
            template_pc, canonical_box = points_utils.cropAndCenterPC(previous_pc, results_bbs[current_frame_id - 1],
                                                                      scale=self.config.model_bb_scale,
                                                                      offset=self.config.model_bb_offset)
        elif "all".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.getModel([frame["pc"] for frame in sequence[:current_frame_id]],
                                                               results_bbs,
                                                               scale=self.config.model_bb_scale,
                                                               offset=self.config.model_bb_offset)
        return template_pc, canonical_box

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
        search_pc_crop = points_utils.generate_subwindow(this_pc, ref_bb,
                                                         scale=self.config.search_bb_scale,
                                                         offset=self.config.search_bb_offset)
        return search_pc_crop, ref_bb

    def prepare_input(self, template_pc, search_pc, template_box, *args, **kwargs):
        """
        construct input dict for evaluating
        :param template_pc:
        :param search_pc:
        :param template_box:
        :return:
        """
        template_points, idx_t = points_utils.regularize_pc(template_pc.points.T, self.config.template_size,
                                                            seed=1)
        search_points, idx_s = points_utils.regularize_pc(search_pc.points.T, self.config.search_size,
                                                          seed=1)
        template_points_torch = torch.tensor(template_points, device=self.device, dtype=torch.float32)
        search_points_torch = torch.tensor(search_points, device=self.device, dtype=torch.float32, requires_grad=True)
        template_bc = points_utils.get_point_to_box_distance(template_points, template_box)
        template_bc_torch = torch.tensor(template_bc, device=self.device, dtype=torch.float32)
        data_dict = {
            'template_points': template_points_torch[None, ...],
            'search_points': search_points_torch[None, ...],
            'points2cc_dist_t': template_bc_torch[None, ...]
        }
        return data_dict

    def evaluate_one_sequence(self, sequence):
        """

        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []
        cd_imperception = []
        hd_imperception = []
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

                # attack
                if getattr(self.config, 'attack', False):
                    adv_path = './adv_examples/' + self.config.net_model + '/' + 'ba3_' + self.config.category_name + '/'
                    os.makedirs(adv_path, exist_ok=True)
                    search_points_clean = data_dict['search_points'][0].clone()
                    np.savetxt(adv_path+f'clean_{self.tracklet_count:04.0f}_{frame_id:04d}.txt', search_points_clean.detach().cpu().numpy(), fmt='%f')

                    # data_dict = self.attack_fgsm(data_dict, this_bb, results_bbs[-1])
                    # data_dict = self.attack_pgd(data_dict, this_bb, results_bbs[-1])
                    # data_dict = self.attack_gauss(data_dict)
                    # data_dict = self.attack_cw(data_dict, this_bb, results_bbs[-1])

                    # data_dict = self.attack_trans(data_dict)
                    data_dict = self.attack_ba(data_dict, this_bb, results_bbs[-1], frame_id)
                    # data_dict = self.attacker.attack_fgsm(data_dict, this_bb, results_bbs[-1])
                    # data_dict = self.attacker.attack_pgd(data_dict, this_bb, results_bbs[-1])
                    # data_dict = self.attacker.attack_cw(data_dict, this_bb, results_bbs[-1])

                    search_points_adv = data_dict['search_points'][0]
                    np.savetxt(adv_path+f'adv_{self.tracklet_count:04.0f}_{frame_id:04d}.txt', search_points_adv.detach().cpu().numpy(), fmt='%f')

                    hd_imperception.append(hausdorff_distance(search_points_adv.detach().cpu().numpy(),
                                                              search_points_clean.detach().cpu().numpy(),
                                                              distance='euclidean'))
                    cd_imperception.append(self.chamferdist(search_points_clean.unsqueeze(0),
                                                            search_points_adv.unsqueeze(0),
                                                            bidirectional=True).detach().cpu().item())

                end_points = self(data_dict)
                estimation_box = end_points['estimation_boxes']
                estimation_boxes_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
                best_box_idx = estimation_boxes_cpu[:, 4].argmax()
                estimation_box_cpu = estimation_boxes_cpu[best_box_idx, 0:4]
                candidate_box = points_utils.getOffsetBB(ref_bb, estimation_box_cpu,
                                                         degrees=self.config.degrees,
                                                         use_z=self.config.use_z,
                                                         limit_box=self.config.limit_box)
                results_bbs.append(candidate_box)
                if getattr(self.config, 'visual', False):
                    don_sampled_ind = end_points['sample_idxs'][:, :128, None].repeat(1, 1, 3).long()
                    don_sampled_points = data_dict['search_points'].gather(dim=1, index=don_sampled_ind)

                    seg_score = end_points['estimation_cla'][:, :, None].sigmoid().repeat(1, 1, 3)
                    seed_vote = end_points['vote_xyz']

                    # # self.logger.experiment.add_mesh('search region', don_sampled_points, config_dict={"Size":10},
                    # #                                 colors=seg_score.sigmoid(), global_step=frame_id)
                    # candidate_box = points_utils.transform_box(candidate_box, ref_bb)
                    # res = vis(data_dict['search_points'][0].cpu().numpy(),
                    #           don_sampled_points[0].cpu().numpy(),
                    #           seed_vote[0].cpu().numpy(),
                    #           candidate_box, this_bb, None,
                    #           seg_score.cpu().numpy())
                    # self.logger.experiment.add_figure('search region', res, frame_id)
                    # # if frame_id > 50: exit()

                    if frame_id != 11:
                        continue
                    fusion_feat = end_points['fusion_feat']
                    fusion_feat = (fusion_feat - torch.min(fusion_feat, dim=1, keepdim=True)[0]) \
                                  / (torch.max(fusion_feat, dim=1, keepdim=True)[0] - torch.min(fusion_feat, dim=1, keepdim=True)[0])
                    for i in range(64):
                        res = vis(data_dict['search_points'][0].cpu().numpy(),
                                  don_sampled_points[0].cpu().numpy(),
                                  seed_vote[0].cpu().numpy(),
                                  candidate_box, this_bb, None,
                                  seg_score.cpu().numpy(),
                                  fusion_feat[:, i, :].cpu().numpy())

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

        return ious, distances, hd_imperception, cd_imperception

    def validation_step(self, batch, batch_idx):
        # if batch_idx == 4:  # for test visual
        if batch_idx != -1:

            sequence = batch[0]  # unwrap the batch with batch size = 1

            ious, distances, hd, cd = self.evaluate_one_sequence(sequence)
            # exit()

            # update metrics
            self.success(torch.tensor(ious, device=self.device))
            self.prec(torch.tensor(distances, device=self.device))
            self.hd(torch.tensor(hd, device=self.device))
            self.cd(torch.tensor(cd, device=self.device))

            self.log('success/test', self.success, on_step=True, on_epoch=True)
            self.log('precision/test', self.prec, on_step=True, on_epoch=True)
        else:
            print("skip: ", batch_idx)

    def validation_epoch_end(self, outputs):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.global_step)
        self.logger.experiment.add_scalars('metrics/HD',
                                           {'HD': self.hd.compute()},
                                           global_step=self.global_step)
        self.logger.experiment.add_scalars('metrics/CD',
                                           {'CD': self.cd.compute()},
                                           global_step=self.global_step)

    def test_step(self, batch, batch_idx):

        # if batch_idx >= 97:  # for test visual
        if batch_idx != -1:

            sequence = batch[0]  # unwrap the batch with batch size = 1

            start_time = time.time()
            ious, distances, hd, cd = self.evaluate_one_sequence(sequence)
            end_start = time.time()

            self.times_tracklet.append((end_start - start_time)/len(sequence))
            # exit()

            self.tracklet_count += 1

            # update metrics
            self.success(torch.tensor(ious, device=self.device))
            self.prec(torch.tensor(distances, device=self.device))
            self.hd(torch.tensor(hd, device=self.device))
            self.cd(torch.tensor(cd, device=self.device))

            self.log('success/test', self.success, on_step=True, on_epoch=True)
            self.log('precision/test', self.prec, on_step=True, on_epoch=True)

        # else:
        #     print("skip: ", batch_idx)

    def test_epoch_end(self, outputs):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.global_step)
        self.logger.experiment.add_scalars('metrics/HD',
                                           {'HD': self.hd.compute()},
                                           global_step=self.global_step)
        self.logger.experiment.add_scalars('metrics/CD',
                                           {'CD': self.cd.compute()},
                                           global_step=self.global_step)

        if self.config.test:
            fps = 1 / (sum(self.times_tracklet) / len(self.times_tracklet))
            print('Running frame per second: {:04.2f}'.format(fps))
            print('HD:{:f}, CD: {:f}'.format(self.hd.compute(), self.cd.compute()))

    def attack_loss(self, end_point, gt_box, prev_box, target_attack=True):
        gt_box_normal = points_utils.transform_box(gt_box, prev_box)
        pred_boxes = end_point['estimation_boxes']
        proposal_center = end_point['center_xyz']

        center = torch.tensor(gt_box_normal.center).cuda()
        dist = torch.sum((proposal_center - center[None, None, :3]) ** 2, dim=-1)
        dist = torch.sqrt(dist + 1e-6)  #

        if target_attack:
            adv_object_label = torch.zeros_like(dist, dtype=torch.float32)
            adv_object_label[dist > 0.3] = 1 # adv label
            loss = F.binary_cross_entropy_with_logits(pred_boxes[:, :, -1], adv_object_label)
            return -loss
        else:
            object_label = torch.zeros_like(dist, dtype=torch.float32)
            object_label[dist < 0.3] = 1 # adv label
            loss = F.binary_cross_entropy_with_logits(pred_boxes[:, :, -1], object_label)
            return loss

    def attack_fgsm(self, data_dict, gt_box, prev_box):
        epsilon = 0.05
        with torch.enable_grad():
            data_dict['search_points'] = data_dict['search_points'].clone().detach().requires_grad_(True)
            candidate_box, end_points = self.run_track(data_dict, prev_box)
            loss = self.attack_loss(end_points, gt_box, prev_box, target_attack=True)
            self.zero_grad()
            loss.backward()
            pertubation = epsilon * torch.sign(data_dict['search_points'].grad)
            data_dict['search_points'] = data_dict['search_points'] + pertubation
        return data_dict

    def attack_pgd(self, data_dict, gt_box, prev_box):
        epsilon = 0.05
        pgd_iter = 10
        with torch.enable_grad():
            data_dict['search_points'] = data_dict['search_points'] + (torch.rand_like(data_dict['search_points']) * 0.1 - 0.05)
            for i in range(pgd_iter):
                data_dict['search_points'] = data_dict['search_points'].clone().detach().requires_grad_(True)
                candidate_box, end_points = self.run_track(data_dict, prev_box)
                loss = self.attack_loss(end_points, gt_box, prev_box, target_attack=True)
                self.zero_grad()
                loss.backward()
                pertubation = epsilon * torch.sign(data_dict['search_points'].grad)
                data_dict['search_points'] = data_dict['search_points'] + pertubation
        return data_dict

    def attack_gauss(self, data_dict):
        epsilon = 0.05
        # pertubation = torch.randn_like(data_dict['search_points'])
        # pertubation = torch.matmul(pertubation, torch.eye(3))
        pertubation = epsilon * torch.randn_like(data_dict['search_points']).sign()
        data_dict['search_points'] = data_dict['search_points'] + pertubation
        return data_dict

    def attack_impluse(self, data_dict):
        p = 0.05  # probability
        N = data_dict['search_points'].size(1)
        rnd = torch.rand(N).cuda()
        mask_low = rnd < p
        mask_high = rnd > (1-p)

        data_dict['search_points'][:, mask_low, :] = data_dict['search_points'][:, mask_low, :] * 0
        data_dict['search_points'][:, mask_high, :] = data_dict['search_points'][:, mask_high, :] * 2
        return data_dict

    def attack_cw(self, data_dict, gt_box, prev_box):
        cw_iter = 10
        c = 1
        lr = 0.01

        pc = data_dict['search_points'].clone().detach()
        # w = 0.05 * torch.ones_like(pc).requires_grad_(True)
        w = 0.1 * torch.rand_like(pc).requires_grad_(True) - 0.05
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

                data_dict['search_points'] = adv_pc
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
            data_dict['search_points'] = pc + w
        return data_dict

    def run_track(self, data_dict, ref_bb):
        end_points = self.forward(data_dict)
        estimation_box = end_points['estimation_boxes']
        estimation_boxes_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
        best_box_idx = estimation_boxes_cpu[:, 4].argmax()
        estimation_box_cpu = estimation_boxes_cpu[best_box_idx, 0:4]
        candidate_box = points_utils.getOffsetBB(ref_bb, estimation_box_cpu, degrees=self.config.degrees,
                                                 use_z=self.config.use_z,
                                                 limit_box=self.config.limit_box)
        return candidate_box, end_points

    def attack_ba(self, data_dict, gt_box, prev_box, frame_id):
        iter_num = 10
        c = 0.5
        lr = 0.01

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

        # adv_path = './adv_examples/' + 'tracker' + '/' + 'ba_Car' + '/'
        # os.makedirs(adv_path, exist_ok=True)
        # np.savetxt(adv_path + f'clean_{self.tracklet_count:04.0f}_{frame_id:04d}.txt',
        #            pc[0, important_mask, :].detach().cpu().numpy(), fmt='%f')
        # np.savetxt(adv_path + f'adv_{self.tracklet_count:04.0f}_{frame_id:04d}.txt',
        #            pc[0, :, :].detach().cpu().numpy(), fmt='%f')

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
            data_dict['search_points'] = pc + w
        return data_dict

    def attack_trans(self, data_dict):
        data_dict['search_points'] = self.attacker.forward_test(data_dict['search_points'],
                                                                data_dict['template_points'])

        return data_dict

    def sample(self, delta, p):
        b, s, n = delta.size()
        only_add_one_mask = torch.from_numpy(np.random.choice([0, 1], size=(b, s, n), p=[1 - p, p])).cuda()

        leave_one_mask = 1 - only_add_one_mask

        only_add_one_perturbation = delta * only_add_one_mask
        leave_one_out_perturbation = delta * leave_one_mask

        return only_add_one_perturbation, leave_one_out_perturbation


