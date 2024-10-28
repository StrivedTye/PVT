""" 
rpn.py
Created by zenn at 2021/5/8 20:55
"""
import torch
from torch import nn
from collections import OrderedDict

from pointnet2.utils import pytorch_utils as pt_utils
from pointnet2.utils.pointnet2_modules import PointnetSAModule
from pointnet2.utils.pointnet2_utils import furthest_point_sample

from models.voxel_utils.voxel.voxelnet import Conv_Middle_layers
from models.voxel_utils.voxel.region_proposal_network import RPN
from models.voxel_utils.voxelization import Voxelization


class P2BVoteNetRPN(nn.Module):

    def __init__(self, feature_channel, vote_channel=256, num_proposal=64, normalize_xyz=False):
        super().__init__()
        self.num_proposal = num_proposal
        self.FC_layer_cla = (
            pt_utils.Seq(feature_channel)
                .conv1d(feature_channel, bn=True)
                .conv1d(feature_channel, bn=True)
                .conv1d(1, activation=None))
        self.vote_layer = (
            pt_utils.Seq(3 + feature_channel)
                .conv1d(feature_channel, bn=True)
                .conv1d(feature_channel, bn=True)
                .conv1d(3 + feature_channel, activation=None))

        self.vote_aggregation = PointnetSAModule(
            radius=0.3,
            nsample=16,
            mlp=[1 + feature_channel, vote_channel, vote_channel, vote_channel],
            use_xyz=True,
            normalize_xyz=normalize_xyz)

        self.FC_proposal = (
            pt_utils.Seq(vote_channel)
                .conv1d(vote_channel, bn=True)
                .conv1d(vote_channel, bn=True)
                .conv1d(3 + 1 + 1, activation=None))

    def forward(self, xyz, feature):
        """

        :param xyz: B,N,3
        :param feature: B,f,N
        :return: B,N,4+1 (xyz,theta,targetnessscore)
        """
        estimation_cla = self.FC_layer_cla(feature).squeeze(1)
        score = estimation_cla.sigmoid()

        xyz_feature = torch.cat((xyz.transpose(1, 2).contiguous(), feature), dim=1)

        offset = self.vote_layer(xyz_feature)
        vote = xyz_feature + offset
        vote_xyz = vote[:, 0:3, :].transpose(1, 2).contiguous()
        vote_feature = vote[:, 3:, :]

        vote_feature = torch.cat((score.unsqueeze(1), vote_feature), dim=1)

        center_xyzs, proposal_features = self.vote_aggregation(vote_xyz, vote_feature, self.num_proposal)
        proposal_offsets = self.FC_proposal(proposal_features)
        estimation_boxes = torch.cat(
            (proposal_offsets[:, 0:3, :] + center_xyzs.transpose(1, 2).contiguous(), proposal_offsets[:, 3:5, :]),
            dim=1)

        estimation_boxes = estimation_boxes.transpose(1, 2).contiguous()
        return estimation_boxes, estimation_cla, vote_xyz, center_xyzs


class P2BVoteNetRPN2(nn.Module):

    def __init__(self, feature_channel, vote_channel=256, num_proposal=64, normalize_xyz=False):
        super().__init__()
        self.num_proposal = num_proposal
        self.FC_layer_cla = (
            pt_utils.Seq(feature_channel)
                .conv1d(feature_channel, bn=True)
                .conv1d(feature_channel, bn=True)
                .conv1d(1, activation=None))
        self.vote_layer = (
            pt_utils.Seq(3 + feature_channel)
                .conv1d(feature_channel, bn=True)
                .conv1d(feature_channel, bn=True)
                .conv1d(3 + feature_channel, activation=None))

        self.vote_aggregation = PointnetSAModule(
            radius=0.3,
            nsample=16,
            mlp=[1 + feature_channel, vote_channel, vote_channel, vote_channel],
            use_xyz=True,
            normalize_xyz=normalize_xyz)

        self.FC_proposal = (
            pt_utils.Seq(vote_channel)
                .conv1d(vote_channel, bn=True)
                .conv1d(vote_channel, bn=True)
                .conv1d(8 + 1, activation=None))

        self.FC_proposal_score = (pt_utils.Seq(vote_channel)
                                  .conv1d(vote_channel, bn=True)
                                  .conv1d(vote_channel, bn=True)
                                  .conv1d(1, activation=None))

    def forward(self, xyz, feature):
        """

        :param xyz: B,N,3
        :param feature: B,f,N
        :return: B,N,4+1 (xyz,theta,targetnessscore)
        """

        estimation_cla = self.FC_layer_cla(feature).squeeze(1)
        score = estimation_cla.sigmoid()

        xyz_feature = torch.cat((xyz.transpose(1, 2).contiguous(), feature), dim=1)

        offset = self.vote_layer(xyz_feature)
        vote = xyz_feature + offset
        vote_xyz = vote[:, 0:3, :].transpose(1, 2).contiguous()
        vote_feature = vote[:, 3:, :]

        vote_feature = torch.cat((score.unsqueeze(1), vote_feature), dim=1)

        center_xyzs, proposal_features = self.vote_aggregation(vote_xyz, vote_feature, self.num_proposal)
        sampled_id = furthest_point_sample(vote_xyz, self.num_proposal).long()  # [B, 64]

        estimation = self.FC_proposal(proposal_features).transpose(1, 2).contiguous()  # [B, 64, 8+1]
        estimation_score = self.FC_proposal_score(proposal_features).transpose(1, 2).contiguous()
        # estimation_score = estimation[:, :, 8:]  # [B, 64, 1]
        estimation_boxes = estimation[:, :, :8]

        return estimation_boxes, estimation_cla, vote_xyz, center_xyzs, estimation_score, sampled_id


class BEVRPNHead(nn.Module):
    def __init__(self, inplanes, num_classes):
        '''
        Args:
            inplanes: input channel
            num_classes: as the name implies
            num_anchors: as the name implies
        '''
        super(BEVRPNHead, self).__init__()
        self.num_classes = num_classes
        self.cls = nn.Sequential(
            nn.Conv2d(inplanes, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)
        )
        self.loc = nn.Sequential(
            nn.Conv2d(inplanes, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1)
        )
        self.z_axis = nn.Sequential(
            nn.Conv2d(inplanes, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, stride=1)
        )

        self.cls[-1].bias.data.fill_(-2.19)
        self.fill_fc_weights(self.loc)
        self.fill_fc_weights(self.z_axis)

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        Args:
            x: [B, inplanes, h, w], input feature
        Return:
            pred_cls: [B, num_anchors, h, w]
            pred_loc: [B, num_anchors*4, h, w]
        '''
        pred_cls = self.cls(x)
        pred_loc = self.loc(x)
        pred_z_axis = self.z_axis(x)
        #(B,9,C)
        return pred_cls, pred_loc, pred_z_axis


class BEVRPN(nn.Module):
    def __init__(self, dim_input=128, num_classes=1):

        super(BEVRPN, self).__init__()
        self.conv = (pt_utils.Seq(dim_input)
                     .conv2d(128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bn=True)
                     .conv2d(128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bn=True)
                     .conv2d(128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bn=True)
                     )

        # self.deconv = nn.Sequential(
        #     OrderedDict(
        #         [('ConvTranspose', nn.ConvTranspose2d(128, 128, 3, 2, 0)),
        #          ('bn', nn.BatchNorm2d(128)),
        #          ('relu', nn.ReLU(inplace=True))]
        #     )
        # )

        self.conv_trans = (pt_utils.Seq(dim_input).conv2d(128, bn=True))
        self.conv_final = (pt_utils.Seq(128*2).conv2d(128, bn=True))

        self.rpn_head = BEVRPNHead(128, num_classes)

    def forward(self, x):
        out1 = self.conv_trans(x)

        out = self.conv(x)
        # out = self.deconv(out)
        out = torch.cat([out1, out], dim=1)

        out = self.conv_final(out)
        pred_cls, pred_loc, pred_z_axis = self.rpn_head(out)

        return pred_cls, pred_loc, pred_z_axis


class VoxelRPN(nn.Module):
    def __init__(self, voxel_area, scene_ground, mode, voxel_size, feat_dim=35):
        super(VoxelRPN, self).__init__()
        self.voxel_size = voxel_size
        self.voxel_area = voxel_area
        self.scene_ground = scene_ground
        self.mode = mode

        self.voxelize = Voxelization(
                                self.voxel_area[0],
                                self.voxel_area[1],
                                self.voxel_area[2],
                                scene_ground=self.scene_ground,
                                mode=self.mode,
                                voxel_size=self.voxel_size)
        self.cml = Conv_Middle_layers(inplanes=feat_dim)
        self.RPN = RPN()

    def forward(self, fusion_xyz_feature, search_xyz):
        voxel_features = self.voxelize(fusion_xyz_feature, search_xyz)
        voxel_features = voxel_features.permute(0, 1, 4, 3, 2).contiguous()
        cml_out = self.cml(voxel_features)
        pred_hm, pred_loc, pred_z_axis = self.RPN(cml_out)
        return pred_hm, pred_loc, pred_z_axis
