import numpy as np
import torch
import os
import copy
from models import p2b, bat
from models import base_model
import torch.nn.functional as F
from datasets import points_utils
from utils.metrics import estimateAccuracy, estimateOverlap
from utils.box_coder import mot_decode

# from models.backbone.pointnet2_trans import Pointnet_Backbone
from models.backbone.pointnet_pp import Pointnet_Backbone
from chamferdist import ChamferDistance
from hausdorff import hausdorff_distance
from utils.graph_spectral import eig_vector

point_size_config = {
    'material': {
        'cls': 'PointsMaterial',
        'size': 0.1
    }
}


class ATTACK3D(base_model.BaseModel):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()
        tracker = self.config.baseline
        baseline = globals()[tracker.lower()].__getattribute__(tracker.upper())
        self.baseline = baseline.load_from_checkpoint(self.config.baseline_ckpt, config=self.config)
        # self.baseline.freeze()

        self.autoencoder = Pointnet_Backbone(self.config.use_fps, self.config.normalize_xyz, return_intermediate=False)
        self.residual = torch.nn.Conv1d(256, 3, kernel_size=1)

        # self.spectral = torch.nn.Conv1d(256, 5, kernel_size=1)

        self.chamferdist = ChamferDistance()

    def configure_optimizers(self):
        betas = getattr(self.config, 'betas', (0.9, 0.999))
        optimizer = torch.optim.Adam([{'params': self.autoencoder.parameters()},
                                      {'params': self.residual.parameters()}],
                                     lr=self.config.lr,
                                     weight_decay=self.config.wd,
                                     betas=tuple(betas), eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def compute_loss(self, output, output_adv):
        estimation_boxes = output['estimation_boxes']  # [B, 64, 4]
        estimation_boxes_adv = output_adv['estimation_boxes']
        proposal_center = output["center_xyz"]  # [B,num_proposal,3]
        proposal_center_adv = output_adv["center_xyz"]
        v = output_adv['graph_egienvector']
        u = output_adv['graph_egienvalue']

        loss_dict = {}

        # score
        max_score_idx = torch.argmax(estimation_boxes[:, :, 4], dim=1, keepdim=True).unsqueeze(2).repeat([1, 1, 3])
        center = torch.gather(proposal_center, dim=1, index=max_score_idx)  # [B, 1, 3]

        dist = torch.norm(proposal_center_adv-center, dim=-1)

        inside_flag = torch.zeros_like(dist, dtype=torch.float32)
        inside_flag[dist <= 0.35] = 1

        outside_flag = torch.zeros_like(dist, dtype=torch.float32)
        outside_flag[dist > 0.35] = 1
        outside_flag_bigger = torch.zeros_like(dist, dtype=torch.float32)
        outside_flag_bigger[dist > 0.65] = 1

        score_pos = torch.max(inside_flag * estimation_boxes_adv[:, :, 4], dim=1)[0]  # B
        score_neg = (torch.max(outside_flag * estimation_boxes_adv[:, :, 4], dim=1)[0]
                     + torch.max(outside_flag_bigger * estimation_boxes_adv[:, :, 4], dim=1)[0]) / 2
        loss_score = torch.mean(score_pos - score_neg)

        self.logger.experiment.add_mesh('Adversarial search region', output_adv['search_points'][:1, :, :],
                                        global_step=self.global_step, config_dict=point_size_config)
        self.logger.experiment.add_mesh('Clean search region', output['search_points'][:1, :, :],
                                        global_step=self.global_step, config_dict=point_size_config)

        # reconstruction
        # loss_recon = F.mse_loss(output_adv['search_points'], output['search_points'])
        loss_recon = 0.5 * self.chamferdist(output_adv['search_points'], output['search_points'], bidirectional=True)

        # Low-level frequency constrian in spectral domain
        # x_gft = torch.einsum('bij,bjk->bik', v.transpose(1, 2), output['search_points'])
        # x_adv_gft = torch.einsum('bij,bjk->bik', v.transpose(1, 2), output_adv['search_points'])
        # mask = torch.ones_like(x_gft)
        # mask[:, :100, :] = 1  # low
        # mask[:, 100:400, :] = 0  # middle
        # mask[:, 400:, :] = 0  # high
        # x_gft = x_gft * mask
        # x_hat = torch.einsum('bij,bjk->bik', v, x_gft)
        # x_adv_hat = torch.einsum('bij,bjk->bik', v, x_adv_gft)
        # loss_graph = F.mse_loss(x_hat, x_adv_hat)

        loss_dict['loss_score'] = loss_score
        loss_dict['loss_recon'] = loss_recon
        # loss_dict['loss_graph'] = loss_graph
        return loss_dict

    def forward(self, input_dict):

        template = input_dict['template_points']
        search = input_dict['search_points']  # B, N, 3

        self.baseline.eval()
        end_points = self.baseline.forward(copy.deepcopy(input_dict))
        end_points['search_points'] = search

        # generate adversarial point clouds by pointNet++
        M, N = template.shape[1], search.shape[1]
        search_xyz, search_feat, sample_idxs = self.autoencoder(search, [N // 2, N // 4, N // 8])

        # learn perturbation by residual
        # search_residual = self.residual(search_feat).transpose(1, 2).contiguous()  # [B, N, 3]
        # search_xyz_adv = search_xyz + search_residual

        # get graph laplacian, L=v'uv
        # v, L, u = eig_vector(search_xyz, 10)  # v: eigien vector; u: eigien value
        v = torch.randn([search.shape[0], search.shape[1], search.shape[1]], device=search.device)
        u = torch.randn([search.shape[0], search.shape[1]], device=search.device)

        u_sort, u_sort_ind = torch.sort(u, dim=1, descending=False)
        v = torch.gather(v, dim=2, index=u_sort_ind.unsqueeze(1).repeat(1, search_xyz.shape[1], 1))

        # Graph Fourier Transformation
        x_gft = torch.einsum('bij,bjk->bik', v.transpose(1, 2), search_xyz)

        # learn graph filter
        search_spectral = self.spectral(search_feat).transpose(1, 2).contiguous()  # [B, N, 5]
        u_sort = u_sort.unsqueeze(-1)
        u_sort_ = torch.cat((torch.ones_like(u_sort), u_sort, u_sort**2, u_sort**3, u_sort**4), dim=-1)  # [B, N, 5]
        graph_filter = search_spectral * u_sort_  # ploynomial, to do: chebeslve
        x_filter = graph_filter.sum(dim=2, keepdim=True).repeat(1, 1, 3) * x_gft  #[B, N, 3]
        search_xyz_adv = torch.einsum('bij,bjk->bik', v, x_filter)

        input_dict['search_points'] = search_xyz_adv
        end_points_adv = self.baseline.forward(input_dict)
        end_points_adv['search_points'] = search_xyz_adv

        end_points_adv['graph_egienvector'] = v
        end_points_adv['graph_egienvalue'] = u_sort

        return end_points, end_points_adv

    def forward_test(self, search, template):
        # generate adversarial point clouds by pointNet++
        M, N = template.shape[1], search.shape[1]
        search_xyz, search_feat, sample_idxs = self.autoencoder(search, [N // 2, N // 4, N // 8])

        search_residual = self.residual(search_feat).transpose(1, 2).contiguous()  # [B, 3, N]
        search_xyz_adv = search_xyz + search_residual

        # # get graph laplacian, L=v'uv
        # v, L, u = eig_vector(search_xyz, 100)  # v: eigien vector; u: eigien value
        # u_sort, u_sort_ind = torch.sort(u, dim=1, descending=False)
        # v = torch.gather(v, dim=2, index=u_sort_ind.unsqueeze(1).repeat(1, search_xyz.shape[1], 1))
        #
        # # Graph Fourier Transformation
        # x_gft = torch.einsum('bij,bjk->bik', v.transpose(1, 2), search_xyz)
        #
        # # learn graph filter
        # search_spectral = self.spectral(search_feat).transpose(1, 2).contiguous()  # [B, N, 5]
        # u_sort = u_sort.unsqueeze(-1)
        # u_sort_ = torch.cat((torch.ones_like(u_sort), u_sort, u_sort**2, u_sort**3, u_sort**4), dim=-1)  # [B, N, 5]
        # graph_filter = search_spectral * u_sort_  # ploynomial, to do: chebeslve
        # x_filter = graph_filter.sum(dim=2, keepdim=True).repeat(1, 1, 3) * x_gft  #[B, N, 3]
        # search_xyz_adv = torch.einsum('bij,bjk->bik', v, x_filter)

        return search_xyz_adv

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

    def attack_fgsm(self, data_dict, gt_box, prev_box, victim=None):
        if victim == "m2track":
            epsilon = torch.tensor([0.05, 0.05, 0.05, 0, 0]).view(1, 1, -1).to(data_dict['points'])
            with torch.enable_grad():
                data_dict['points'] = data_dict['points'].clone().detach().requires_grad_(True)
                data_dict['search_points'] = data_dict['points'][:, 1024:, :3]
                data_dict['template_points'] = data_dict['points'][:, :1024, :3]
                data_dict['points2cc_dist_t'] = data_dict['candidate_bc'][:, :1024, :]
                end_points = self.baseline.forward(data_dict)
                candidate_box, end_points = self.run_track(end_points, prev_box)
                loss = self.attack_loss(end_points, gt_box, prev_box, target_attack=True)
                self.zero_grad()
                loss.backward()
                pertubation = epsilon * torch.sign(data_dict['points'].grad)
                data_dict['points'] = data_dict['points'] + pertubation
        else:
            epsilon = 0.05
            with torch.enable_grad():
                data_dict['search_points'] = data_dict['search_points'].clone().detach().requires_grad_(True)
                end_points = self.baseline.forward(data_dict)
                candidate_box, end_points = self.run_track(end_points, prev_box)
                loss = self.attack_loss(end_points, gt_box, prev_box, target_attack=True)
                self.zero_grad()
                loss.backward()
                pertubation = epsilon * torch.sign(data_dict['search_points'].grad)
                data_dict['search_points'] = data_dict['search_points'] + pertubation

        return data_dict

    def attack_pgd(self, data_dict, gt_box, prev_box, victim=None):
        if victim == "m2track":
            epsilon = torch.tensor([0.05, 0.05, 0.05, 0, 0]).view(1, 1, -1).to(data_dict['points'])
            pgd_iter = 10
            with torch.enable_grad():
                data_dict['points'] = data_dict['points'] + (torch.rand_like(data_dict['points']) * 0.1 - 0.05)
                for i in range(pgd_iter):
                    data_dict['points'] = data_dict['points'].clone().detach().requires_grad_(True)
                    data_dict['search_points'] = data_dict['points'][:, 1024:, :3]
                    data_dict['template_points'] = data_dict['points'][:, :1024, :3]
                    data_dict['points2cc_dist_t'] = data_dict['candidate_bc'][:, :1024, :]
                    end_points = self.baseline.forward(data_dict)
                    candidate_box, end_points = self.run_track(end_points, prev_box)

                    loss = self.attack_loss(end_points, gt_box, prev_box, target_attack=True)
                    self.zero_grad()
                    loss.backward()
                    pertubation = epsilon * torch.sign(data_dict['points'].grad)
                    data_dict['points'] = data_dict['points'] + pertubation
        else:
            epsilon = 0.05
            pgd_iter = 10
            with torch.enable_grad():
                data_dict['search_points'] = data_dict['search_points'] + (torch.rand_like(data_dict['search_points']) * 0.1 - 0.05)
                for i in range(pgd_iter):
                    data_dict['search_points'] = data_dict['search_points'].clone().detach().requires_grad_(True)
                    end_points = self.baseline.forward(data_dict)
                    candidate_box, end_points = self.run_track(end_points, prev_box)
                    loss = self.attack_loss(end_points, gt_box, prev_box, target_attack=True)
                    self.zero_grad()
                    loss.backward()
                    pertubation = epsilon * torch.sign(data_dict['search_points'].grad)
                    data_dict['search_points'] = data_dict['search_points'] + pertubation
        return data_dict

    def attack_cw(self, data_dict, gt_box, prev_box, victim=None):
        if victim == "m2track":
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

                    data_dict['search_points'] = adv_pc[:, 1024:, :3]
                    data_dict['template_points'] = pc[:, :1024, :3]
                    data_dict['points2cc_dist_t'] = data_dict['candidate_bc'][:, :1024, :]

                    end_points = self.baseline.forward(data_dict)
                    # data_dict['points'] = torch.cat([adv_pc, data_dict['points'][:, :, 3:]], dim=2)
                    candidate_box, end_points = self.run_track(end_points, prev_box)

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
        else:
            cw_iter = 10
            c = 1
            lr = 0.01

            pc = data_dict['search_points'].clone().detach()
            w = 0.05 * torch.ones_like(pc).requires_grad_(True)
            w = torch.nn.Parameter(w)
            prev_cost = 1e6

            optimizer = torch.optim.Adam([w], lr=lr)
            with torch.enable_grad():
                for i in range(cw_iter):

                    # adversarial example
                    adv_pc = pc + w.clone()

                    # calculate loss
                    current_L2 = F.mse_loss(adv_pc, pc, reduction='none')
                    L2_loss = current_L2.sum(dim=(1, 2)).mean()

                    data_dict['search_points'] = adv_pc
                    end_points = self.baseline.forward(data_dict)
                    candidate_box, end_points = self.run_track(end_points, prev_box)

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
                data_dict['search_points'] = pc + w.clone()
        return data_dict

    def run_track(self, end_points, ref_bb, tracker='p2b'):
        if tracker.lower() in ['p2b', 'bat']:
            estimation_box = end_points['estimation_boxes']
            estimation_boxes_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
            best_box_idx = estimation_boxes_cpu[:, 4].argmax()
            estimation_box_cpu = estimation_boxes_cpu[best_box_idx, 0:4]
            candidate_box = points_utils.getOffsetBB(ref_bb, estimation_box_cpu, degrees=self.config.degrees,
                                                     use_z=self.config.use_z,
                                                     limit_box=self.config.limit_box)
        elif tracker.lower() == 'm2track':
            estimation_box = end_points['estimation_boxes']
            estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
            if len(estimation_box.shape) == 3:
                best_box_idx = estimation_box_cpu[:, 4].argmax()
                estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]

            candidate_box = points_utils.getOffsetBB(ref_bb, estimation_box_cpu,
                                                     degrees=self.config.degrees,
                                                     use_z=self.config.use_z,
                                                     limit_box=self.config.limit_box)
        else:
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
        return candidate_box, end_points

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

        tracker = self.config.baseline.lower()
        for frame_id in range(len(sequence)):  # tracklet
            this_bb = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
                last_preturb = np.zeros_like(sequence[frame_id]["pc"])
            else:
                # construct input dict
                if tracker == 'm2track':
                    data_dict, ref_bb = self.baseline.build_input_dict(sequence, frame_id, results_bbs)
                    end_points, end_points_adv = self.forward(data_dict)
                else:
                    search_pc_crop, ref_bb = self.baseline.generate_search_area(sequence, frame_id, results_bbs)
                    template_pc, canonical_box = self.baseline.generate_template(sequence, frame_id, results_bbs)
                    data_dict = self.baseline.prepare_input(template_pc, search_pc_crop, canonical_box)
                    end_points, end_points_adv = self.forward(data_dict)

                    # adv_path = './adv_examples/' + self.config.net_model + '/' + 'score_recon_' + self.config.category_name + '/'
                    # os.makedirs(adv_path, exist_ok=True)
                    search_points_clean = end_points['search_points'][0]
                    # np.savetxt(adv_path + f'clean_{self.tracklet_count:04.0f}_{frame_id:04d}.txt',
                    #            search_points_clean.detach().cpu().numpy(), fmt='%f')

                    search_points_adv = end_points_adv['search_points'][0]
                    # np.savetxt(adv_path + f'adv_{self.tracklet_count:04.0f}_{frame_id:04d}.txt',
                    #            search_points_adv.detach().cpu().numpy(), fmt='%f')
                    hd_imperception.append(hausdorff_distance(search_points_adv.detach().cpu().numpy(),
                                                              search_points_clean.detach().cpu().numpy(),
                                                              distance='euclidean'))
                    cd_imperception.append(self.chamferdist(search_points_clean.unsqueeze(0),
                                                            search_points_adv.unsqueeze(0),
                                                            bidirectional=True).detach().cpu().item())

                candidate_box, end_points_adv = self.run_track(end_points_adv, ref_bb, tracker)
                results_bbs.append(candidate_box)

            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)
        return ious, distances, hd_imperception, cd_imperception
        # return self.baseline.evaluate_one_sequence(sequence)

    def training_step(self, batch, batch_idx):
        end_points, end_points_adv = self.forward(batch)
        loss_dict = self.compute_loss(end_points, end_points_adv)
        loss = loss_dict['loss_score'] * self.config.score_weight \
               + loss_dict['loss_recon'] * self.config.recon_weight
               # + loss_dict['loss_graph'] * self.config.graph_weight

        self.log('loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_score/train', loss_dict['loss_score'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_recon/train', loss_dict['loss_recon'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        # self.log('loss_graph/train', loss_dict['loss_graph'].item(), on_step=True, on_epoch=True, prog_bar=True,
        #          logger=False)

        self.logger.experiment.add_scalars('loss', {'loss_total': loss.item(),
                                                    'loss_score': loss_dict['loss_score'].item(),
                                                    'loss_recon': loss_dict['loss_recon'].item(),
                                                    # 'loss_graph': loss_dict['loss_graph'].item()
                                                    },
                                           global_step=self.global_step)
        return loss