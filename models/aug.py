import random
import torch
from torch import nn
from models import base_model
import torch.nn.functional as F
from models import p2b, bat
from copy import deepcopy


class AUG(base_model.BaseModel):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()
        baseline = self.config.baseline
        self.baseline = globals()[baseline.lower()].__getattribute__(baseline.upper())(self.config)
        self.augmentor = Augmentor(dim=512)

    def configure_optimizers(self):
        opt_d = torch.optim.Adam(self.baseline.parameters(), lr=self.config.lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.augmentor.parameters(), lr=self.config.lr, betas=(0.9, 0.999))
        lr_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=100,
                                               gamma=self.config.lr_decay_rate)
        lr_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=self.config.lr_decay_step,
                                               gamma=self.config.lr_decay_rate)
        return [opt_g, opt_d], [lr_g, lr_d]

    def compute_loss(self, data, output):
        loss_dict = self.baseline.compute_loss(data, output)
        loss = loss_dict['loss_objective'] * self.config.objectiveness_weight \
               + loss_dict['loss_box'] * self.config.box_weight \
               + loss_dict['loss_seg'] * self.config.seg_weight \
               + loss_dict['loss_vote'] * self.config.vote_weight \
               + loss_dict['loss_bc'] * self.config.bc_weight
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):

        if self.current_epoch < self.config.warm_up:
            if optimizer_idx == 1:
                loss_ori = self.compute_loss(batch, self.baseline(batch))
                return loss_ori
        else:
            # augment step
            if optimizer_idx == 0:
                aug_pt = self.augmentor(deepcopy(batch))
                loss_aug = self.compute_loss(aug_pt, self.baseline(aug_pt))

                loss_ori = self.compute_loss(batch, self.baseline(batch))
                rho = (1.2 + 0.1 * torch.tensor(self.current_epoch % 10)).cuda()
                loss_adv = torch.abs(1 - torch.exp(loss_aug - rho * loss_ori.detach()))
                # print(loss_aug, loss_adv)
                loss = loss_adv

                # self.logger.experiment.add_mesh('search_region', aug_pt["search_points"],
                #                                 config_dict={"Size": 20}, global_step=self.global_step)
                # self.logger.experiment.add_mesh('template', aug_pt["template_points"],
                #                                 config_dict={"Size": 20}, global_step=self.global_step)

            if optimizer_idx == 1:
                aug_pt = self.augmentor(deepcopy(batch))

                loss_aug = self.compute_loss(aug_pt, self.baseline(aug_pt))

                loss_ori = self.compute_loss(batch, self.baseline(batch))
                loss = loss_aug + loss_ori

            return loss

    def evaluate_one_sequence(self, sequence):
        return self.baseline.evaluate_one_sequence(sequence)


class AugmentorRotation(nn.Module):
    def __init__(self, dim=1024):
        super(AugmentorRotation, self).__init__()
        self.fc1 = nn.Linear(dim + dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        # convert quaternion to rotation matrix
        x, s = batch_quat_to_rotmat(x)
        x = x.view(-1, 3, 3)
        s = s.view(B, 1, 1)
        return x, None


class AugmentorDisplacement(nn.Module):
    def __init__(self, dim=1024):
        super(AugmentorDisplacement, self).__init__()

        self.conv1 = nn.Conv1d(dim + 64, 1024, 1)

        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 64, 1)
        self.conv4 = nn.Conv1d(64, 3, 1)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        return x


class Augmentor(nn.Module):
    def __init__(self, dim=1024, in_dim=3, use_rot=False, use_dis=True):
        super(Augmentor, self).__init__()
        self.dim = dim
        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(dim)

        self.use_rot = use_rot
        self.use_dis = use_dis

        self.rot = AugmentorRotation(self.dim) if self.use_rot else None
        self.dis = AugmentorDisplacement(self.dim) if self.use_dis else None

    def forward(self, data_dict):
        template = data_dict['template_points']
        search = data_dict['search_points']  # b, n, 3
        seg_label = data_dict['seg_label']
        B, N, _ = template.size()

        # ind_fg = torch.nonzero(seg_label)
        # ind_fg_repair = []
        # for b in range(B):
        #     cur_ind_fg = ind_fg[ind_fg[:, 0] == b]
        #     num_fg = cur_ind_fg.size(0)  #(ind_fg[:, 0] == b).sum()
        #     if num_fg == 0:
        #         sample_id = torch.zeros(512).to(num_fg.device).long()
        #         ind_fg_repair.append(sample_id)
        #     else:
        #         sample_id = torch.arange(512).to(num_fg.device) % num_fg
        #         ind_fg_repair.append(cur_ind_fg[sample_id][:, 1])
        # ind_fg_repair = torch.stack(ind_fg_repair).unsqueeze(-1)  # b, 512

        # search_fg = torch.gather(search, 1, ind_fg_repair.repeat([1, 1, 3]))
        # # search_fg = search[ind_fg_repair[:, 0], ind_fg_repair[:, 1], :].view(B, 512, 3)

        # noise = 0.02 * torch.randn(B, N).cuda()
        # aug_template = self.transformation(template, noise)
        # aug_search_fg = self.transformation(search_fg, noise)

        noise = 0.02 * torch.randn(B, N).cuda()
        rot_search, disp_search = self.transformation(search, noise)
        rot_template, disp_template = self.transformation(template, noise)

        if random.uniform(0, 1) > 0.0:
            aug_search = rot_search + disp_search * seg_label.unsqueeze(-1)
        else:
            aug_search = rot_search

        if random.uniform(0, 1) > 0.0:
            aug_template = rot_template + disp_template
        else:
            aug_template = rot_template

        # search_list = []
        # for b in range(B):
        #     cur_search = search[b]
        #     cur_search[seg_label[b].long()] = aug_search[b][seg_label[b].long()]
        #     search_list.append(cur_search)

        data_dict['template_points'] = aug_template
        data_dict['search_points'] = aug_search
        return data_dict

    def transformation(self, pt, noise):
        B, N, C = pt.size()
        raw_pt = pt.transpose(1, 2).contiguous()  # b, 3, n

        x = F.relu(self.bn1(self.conv1(raw_pt)))
        x = F.relu(self.bn2(self.conv2(x)))
        feat_point = x   # [b, 64, n]
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, 2, keepdim=True)[0]  # [B, 512, 1]

        if self.use_rot:
            feat_r = x.view(-1, self.dim)
            feat_r = torch.cat([feat_r, noise], 1)
            rotation, scale = self.rot(feat_r)  # [b, 3, 3]

            pt = torch.bmm(pt, rotation)

        feat_d = x.view(-1, self.dim, 1).repeat(1, 1, N)
        noise_d = noise.view(B, -1, 1).repeat(1, 1, N)
        # feat_d = torch.cat([feat_point, feat_d, noise_d], 1)
        feat_d = torch.cat([feat_point, feat_d], 1)
        displacement = self.dis(feat_d).transpose(1, 2).contiguous()  # [b, n, 3]

        return pt, displacement


def batch_quat_to_rotmat(q, out=None):
    B = q.size(0)

    if out is None:
        out = q.new_empty(B, 3, 3)

    # 2 / squared quaternion 2-norm
    len = torch.sum(q.pow(2), 1)
    s = 2 / len

    s_ = torch.clamp(len, 2.0 / 3.0, 3.0 / 2.0)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = (1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s))  # .mul(s_)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = (1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s))  # .mul(s_)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = (1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s))  # .mul(s_)

    return out, s_

