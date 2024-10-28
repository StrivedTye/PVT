""" 
Created by tye at 2021/11/8
"""
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from pointnet2.utils.pointnet2_utils import furthest_point_sample


def get_block(norm_type,
              inplanes,
              planes,
              stride=1,
              dilation=1,
              downsample=None,
              bn_momentum=0.1,
              D=3):
    if norm_type == 'BN':
        return BasicBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
    elif norm_type == 'IN':
        return BasicBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
    else:
        raise ValueError(f'Type {norm_type}, not defined')


def get_norm(norm_type, num_feats, bn_momentum=0.05, D=-1):
    if norm_type == 'BN':
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
    elif norm_type == 'IN':
        return ME.MinkowskiInstanceNorm(num_feats, dimension=D)
    else:
        raise ValueError(f'Type {norm_type}, not defined')


class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = 'BN'

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 D=3):
        super(BasicBlockBase, self).__init__()

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dimension=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
        self.conv2 = ME.MinkowskiConvolution(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            dimension=D)
        self.norm2 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = MEF.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = MEF.relu(out)

        return out


class BasicBlockBN(BasicBlockBase):
    NORM_TYPE = 'BN'


class BasicBlockIN(BasicBlockBase):
    NORM_TYPE = 'IN'


# ResUNet
class ResUNet2(ME.MinkowskiNetwork):
    NORM_TYPE = None
    BLOCK_NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 32, 64, 64, 128]

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(self,
                 in_channels=3,
                 out_channels=32,
                 stride=1,
                 bn_momentum=0.1,
                 normalize_feature=None,
                 conv1_kernel_size=5,
                 D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        NORM_TYPE = self.NORM_TYPE
        BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
        CHANNELS = self.CHANNELS
        TR_CHANNELS = self.TR_CHANNELS
        self.normalize_feature = normalize_feature
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[1],
            kernel_size=conv1_kernel_size,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.block1 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv4 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[3],
            out_channels=CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv4_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[4],
            out_channels=TR_CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4_tr = get_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv3_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[3] + TR_CHANNELS[4],
            out_channels=TR_CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3_tr = get_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[2] + TR_CHANNELS[3],
            out_channels=TR_CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2_tr = get_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv1_tr = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1] + TR_CHANNELS[2],
            out_channels=TR_CHANNELS[1],
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)

        # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.final = ME.MinkowskiConvolution(
            in_channels=TR_CHANNELS[1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=D)

    def forward(self, x):
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out_s1 = self.block1(out_s1)
        out = MEF.relu(out_s1)

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out_s2 = self.block2(out_s2)
        out = MEF.relu(out_s2)

        out_s4 = self.conv3(out)
        out_s4 = self.norm3(out_s4)
        out_s4 = self.block3(out_s4)
        out = MEF.relu(out_s4)

        out_s8 = self.conv4(out)
        out_s8 = self.norm4(out_s8)
        out_s8 = self.block4(out_s8)
        out = MEF.relu(out_s8)

        out = self.conv4_tr(out)
        out = self.norm4_tr(out)
        out = self.block4_tr(out)
        out_s4_tr = MEF.relu(out)

        out = ME.cat(out_s4_tr, out_s4)

        out = self.conv3_tr(out)
        out = self.norm3_tr(out)
        out = self.block3_tr(out)
        out_s2_tr = MEF.relu(out)

        out = ME.cat(out_s2_tr, out_s2)

        out = self.conv2_tr(out)
        out = self.norm2_tr(out)
        out = self.block2_tr(out)
        out_s1_tr = MEF.relu(out)

        out = ME.cat(out_s1_tr, out_s1)
        out = self.conv1_tr(out)
        out = MEF.relu(out)
        out = self.final(out)

        if self.normalize_feature:
            return ME.SparseTensor(
                out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
                coordinate_map_key=out.coordinate_map_key,
                coordinate_manager=out.coordinate_manager)
        else:
            return out


class ResUNetBN2(ResUNet2):
    NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2D(ResUNet2):
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
    NORM_TYPE = 'BN'
    CHANNELS = [None, 128, 128, 128, 256]
    TR_CHANNELS = [None, 64, 128, 128, 128]


class SparseConvBackbone(nn.Module):

    def __init__(self,
                 input_feature_dim=3,
                 output_feature_dim=192,
                 conv1_kernel_size=5,
                 checkpoint=None):
        super().__init__()
        self.net = ResUNetBN2C(input_feature_dim, output_feature_dim,
                               conv1_kernel_size=conv1_kernel_size,
                               bn_momentum=0.05, normalize_feature=None,
                               D=3)
        if checkpoint != 'None':
            self.net.load_state_dict(torch.load(checkpoint)['state_dict'])

    def forward(self, points, coords, feats, inds, num_seed=128, use_feat_norm=False, end_points=None):
        coords = coords.to(device=points.device)
        feats = feats.to(device=points.device)
        inds = inds.to(device=points.device)

        if end_points is None:
            end_points = {}

        inputs = ME.SparseTensor(feats, coords, device=points.device)
        outputs = self.net(inputs)
        features = outputs.F  # [B*N, C]

        # randomly down-sample to num_seed points & create batches
        bsz, num_points, _ = points.size()
        points_v = points.view(-1, 3)
        batch_ids = coords[:, 0]

        #  the actual id in original cloud, which is sampled point in the voxel
        voxel_ids = inds + batch_ids * num_points

        sampled_inds, sampled_feartures, sampled_points, sampled_coords = [], [], [], []
        quantized_points, quantized_points_features = [], []
        num_quantized_points = []
        for b in range(bsz):

            if use_feat_norm:
                cur_feat = features[batch_ids == b]  # [N, C]
                feat_norm = torch.norm(cur_feat, p=2, dim=1)
                true_num_points = len(feat_norm)
                if true_num_points < num_seed:
                    sampled_id = torch.topk(feat_norm, k=true_num_points).indices
                    sampled_id_repair = sampled_id[0].repeat(num_seed - true_num_points)
                    sampled_id = torch.cat([sampled_id, sampled_id_repair], dim=0)
                else:
                    sampled_id = torch.topk(feat_norm, k=num_seed).indices
            else:
                cur_point_v = points_v[voxel_ids[batch_ids == b]]
                # num_seed = self.num_seed if cur_point_v.shape[0] > self.num_seed else cur_point_v.shape[0]
                sampled_id = furthest_point_sample(cur_point_v.unsqueeze(0),
                                                   num_seed).squeeze(0).long()

            sampled_inds.append(inds[batch_ids == b][sampled_id])  # the index of 128 sampled points
            sampled_feartures.append(features[batch_ids == b][sampled_id])
            sampled_points.append(points_v[voxel_ids[batch_ids == b]][sampled_id])
            sampled_coords.append(outputs.C[batch_ids == b][sampled_id])

            # num_quantized_points.append(len(voxel_ids[batch_ids == b]))
            # quantized_points.append(points_v[voxel_ids[batch_ids == b]])
            # quantized_points_features.append(features[batch_ids == b])

        # max_num_q_p = max(num_quantized_points)
        # for i, cur_q_p in enumerate(quantized_points):
        #     if len(cur_q_p) < max_num_q_p:
        #         cur_num_q_p = len(cur_q_p)
        #         ids = torch.randint(0, cur_num_q_p, size=[max_num_q_p-cur_num_q_p])
        #         quantized_points[i] = torch.cat([cur_q_p, cur_q_p[ids]], 0)
        #
        #         cur_q_p_feat = quantized_points_features[i]
        #         quantized_points_features[i] = torch.cat([cur_q_p_feat, cur_q_p_feat[ids]], 0)

        end_points['fp2_features'] = torch.stack(sampled_feartures, 0).transpose(1, 2).contiguous()
        end_points['fp2_xyz'] = torch.stack(sampled_points, 0)
        end_points['fp2_inds'] = torch.stack(sampled_inds, 0).cuda()  # collen_fn did not transform it to cuda
        end_points['fp2_coords'] = torch.stack(sampled_coords, 0)
        # end_points['in_xyz'] = torch.stack(quantized_points, 0)
        # end_points['in_feature'] = torch.stack(quantized_points_features, 0).transpose(1, 2)

        # return end_points['in_xyz'], end_points['in_feature'], \
        #        end_points['fp2_xyz'], end_points['fp2_features'], \
        #        end_points['fp2_inds']

        return end_points['fp2_xyz'], end_points['fp2_features'], end_points['fp2_inds'], end_points['fp2_coords']
