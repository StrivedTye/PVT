# Created by zenn at 2021/5/6
import torch
import os
import copy
import numpy as np
from pyquaternion import Quaternion
import nuscenes.utils.geometry_utils

from datasets.data_classes import PointCloud, Box
from scipy.spatial.distance import cdist

# np.random.seed(666)

def random_choice(num_samples, size, replacement=False, seed=None):
    if seed is not None:
        generator = torch.random.manual_seed(seed)
    else:
        generator = None
    return torch.multinomial(
        torch.ones((size), dtype=torch.float32),
        num_samples=num_samples,
        replacement=replacement,
        generator=generator
    )


def regularize_pc(points, sample_size, seed=None):
    # random sampling from points
    num_points = points.shape[0]
    new_pts_idx = None
    rng = np.random if seed is None else np.random.default_rng(seed)
    if num_points > 2:
        if num_points != sample_size:
            new_pts_idx = rng.choice(num_points, size=sample_size, replace=sample_size > num_points)
            # new_pts_idx = random_choice(num_samples=sample_size, size=num_points,
            #                             replacement=sample_size > num_points, seed=seed).numpy()
        else:
            new_pts_idx = np.arange(num_points)
    if new_pts_idx is not None:
        points = points[new_pts_idx, :]
    else:
        points = np.zeros((sample_size, 3), dtype='float32')
    return points, new_pts_idx


def getOffsetBB(box, offset, degrees=True, use_z=False, limit_box=True, limit_box_abs=False):
    rot_quat = Quaternion(matrix=box.rotation_matrix)
    trans = np.array(box.center)

    new_box = copy.deepcopy(box)

    if len(offset) == 7:

        wlh = offset[[4, 3, 5]] # width, length, height
        offset = offset[[0, 1, 2, 6]]

        if np.abs(wlh[0] - new_box.wlh[0]) > 0.5:
            wlh[0] = new_box.wlh[0]
        if np.abs(wlh[1] - new_box.wlh[1]) > 0.5:
            wlh[1] = new_box.wlh[1]
        if np.abs(wlh[2] - new_box.wlh[2]) > 0.5:
            wlh[2] = new_box.wlh[2]
        new_box = Box(trans, wlh, rot_quat)

    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)
    if len(offset) == 3:
        use_z = False

    # REMOVE TRANSfORM
    if degrees:
        if len(offset) == 3:
            new_box.rotate(
                Quaternion(axis=[0, 0, 1], degrees=offset[2]))
        elif len(offset) == 4:
            new_box.rotate(
                Quaternion(axis=[0, 0, 1], degrees=offset[3]))
    else:
        if len(offset) == 3:
            new_box.rotate(
                Quaternion(axis=[0, 0, 1], radians=offset[2]))
        elif len(offset) == 4:
            new_box.rotate(
                Quaternion(axis=[0, 0, 1], radians=offset[3]))

    if limit_box and not limit_box_abs: # for test
        if offset[0] > new_box.wlh[0]:
            offset[0] = np.random.uniform(-1, 1)

        if offset[1] > min(new_box.wlh[1], 2):
            offset[1] = np.random.uniform(-1, 1)

        if use_z and offset[2] > min(new_box.wlh[2], 0.5):
            offset[2] = np.random.uniform(-0.5, 0.5)

    if limit_box_abs:  # for train
        if np.abs(offset[0]) > min(new_box.wlh[1], 2):
            offset[0] = np.random.uniform(-1, 1)

        if np.abs(offset[1]) > min(new_box.wlh[0], 2):
            offset[1] = np.random.uniform(-1, 1)

        if use_z and np.abs(offset[2]) > min(new_box.wlh[2], 0.5):
            offset[2] = np.random.uniform(-0.5, 0.5)

    if use_z:
        new_box.translate(np.array([offset[0], offset[1], offset[2]]))
    else:
        new_box.translate(np.array([offset[0], offset[1], 0]))

    # APPLY PREVIOUS TRANSFORMATION
    new_box.rotate(rot_quat)
    new_box.translate(trans)
    return new_box


def getModel(PCs, boxes, offset=0, scale=1.0, normalize=False):
    """center and merge the object pcs in boxes"""
    if len(PCs) == 0:
        return PointCloud(np.ones((3, 0)))
    points = [np.ones((PCs[0].points.shape[0], 0), dtype='float32')]
    for PC, box in zip(PCs, boxes):
        cropped_PC, new_box = cropAndCenterPC(PC, box, offset=offset, scale=scale, normalize=normalize)
        # try:
        if cropped_PC.nbr_points() > 0:
            points.append(cropped_PC.points)

    PC = PointCloud(np.concatenate(points, axis=1))
    return PC, new_box


def cropAndCenterPC(PC, box, offset=0, scale=1.0, normalize=False):
    """
    crop and center the pc using the given box
    """
    new_PC = crop_pc_axis_aligned(PC, box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(box)

    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    new_PC.translate(trans)
    new_box.translate(trans)
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    new_PC = crop_pc_axis_aligned(new_PC, new_box, offset=offset, scale=scale)

    if normalize:
        new_PC.normalize(box.wlh)
    return new_PC, new_box


def get_point_to_box_distance(pc, box):
    """
    generate the BoxCloud for the given pc and box
    :param pc: Pointcloud object or numpy array
    :param box:
    :return:
    """
    if isinstance(pc, PointCloud):
        points = pc.points.T  # N,3
    else:
        points = pc  # N,3
        assert points.shape[1] == 3
    box_corners = box.corners()  # 3,8
    box_centers = box.center.reshape(-1, 1)  # 3,1
    box_points = np.concatenate([box_centers, box_corners], axis=1)  # 3,9
    points2cc_dist = cdist(points, box_points.T)  # N,9
    return points2cc_dist


def crop_pc_axis_aligned(PC, box, offset=0, scale=1.0, return_mask=False, limit_area=None):
    """
    crop the pc using the box in the axis-aligned manner
    """
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    if limit_area is not None:
        max_area = np.max(limit_area, 1)
        min_area = np.min(limit_area, 1)
        x_area_max = PC.points[0, :] < max_area[0]
        x_area_min = PC.points[0, :] > min_area[0]
        y_area_max = PC.points[1, :] < max_area[1]
        y_area_min = PC.points[1, :] > min_area[1]
        z_area_max = PC.points[2, :] < max_area[2]
        z_area_min = PC.points[2, :] > min_area[2]

        close = np.logical_and(close, x_area_min)
        close = np.logical_and(close, x_area_max)
        close = np.logical_and(close, y_area_min)
        close = np.logical_and(close, y_area_max)
        close = np.logical_and(close, z_area_min)
        close = np.logical_and(close, z_area_max)

    new_PC = PointCloud(PC.points[:, close])
    if return_mask:
        return new_PC, close
    return new_PC


def crop_pc_oriented(PC, box, offset=0, scale=1.0, return_mask=False):
    """
    crop the pc using the exact box.
    slower than 'crop_pc_axis_aligned' but more accurate
    """

    box_tmp = copy.deepcopy(box)
    new_PC = PointCloud(PC.points.copy())
    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    # align data
    new_PC.translate(trans)
    box_tmp.translate(trans)
    new_PC.rotate(rot_mat)
    box_tmp.rotate(Quaternion(matrix=rot_mat))

    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = new_PC.points[0, :] < maxi[0]
    x_filt_min = new_PC.points[0, :] > mini[0]
    y_filt_max = new_PC.points[1, :] < maxi[1]
    y_filt_min = new_PC.points[1, :] > mini[1]
    z_filt_max = new_PC.points[2, :] < maxi[2]
    z_filt_min = new_PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(new_PC.points[:, close])

    # transform back to the original coordinate system
    new_PC.rotate(np.transpose(rot_mat))
    new_PC.translate(-trans)
    if return_mask:
        return new_PC, close
    return new_PC


def generate_subwindow(pc, sample_bb, scale, offset=2, oriented=True, limit_area=None):
    """
    generating the search area using the sample_bb

    :param pc:
    :param sample_bb:
    :param scale:
    :param offset:
    :param oriented: use oriented or axis-aligned cropping
    :return:
    """
    rot_mat = np.transpose(sample_bb.rotation_matrix)
    trans = -sample_bb.center
    if oriented:
        new_pc = PointCloud(pc.points.copy())
        box_tmp = copy.deepcopy(sample_bb)

        # transform to the coordinate system of sample_bb
        new_pc.translate(trans)
        box_tmp.translate(trans)
        new_pc.rotate(rot_mat)
        box_tmp.rotate(Quaternion(matrix=rot_mat))
        new_pc = crop_pc_axis_aligned(new_pc, box_tmp, scale=scale, offset=offset, limit_area=limit_area)

    else:
        new_pc = crop_pc_axis_aligned(pc, sample_bb, scale=scale, offset=offset, limit_area=limit_area)

        # transform to the coordinate system of sample_bb
        new_pc.translate(trans)
        new_pc.rotate(rot_mat)

    return new_pc


def transform_box(box, ref_box):
    box = copy.deepcopy(box)
    box.translate(-ref_box.center)
    box.rotate(Quaternion(matrix=ref_box.rotation_matrix.T))
    return box


def get_in_box_mask(PC, box):
    """check which points of PC are inside the box"""

    if not isinstance(PC, PointCloud):
        PC = PointCloud(PC)

    box_tmp = copy.deepcopy(box)
    new_PC = PointCloud(PC.points.copy())
    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    # align data
    new_PC.translate(trans)
    box_tmp.translate(trans)
    new_PC.rotate(rot_mat)
    box_tmp.rotate(Quaternion(matrix=rot_mat))
    maxi = np.max(box_tmp.corners(), 1)
    mini = np.min(box_tmp.corners(), 1)

    x_filt_max = new_PC.points[0, :] < maxi[0]
    x_filt_min = new_PC.points[0, :] > mini[0]
    y_filt_max = new_PC.points[1, :] < maxi[1]
    y_filt_min = new_PC.points[1, :] > mini[1]
    z_filt_max = new_PC.points[2, :] < maxi[2]
    z_filt_min = new_PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)
    return close

def apply_transform(in_box_pc, box, translation, rotation, flip_x, flip_y, rotation_axis=(0, 0, 1)):
    """
    Apply transformation to the box and its pc insides. pc should be inside the given box.
    :param in_box_pc: PointCloud object
    :param box: Box object
    :param flip_y: boolean
    :param flip_x: boolean
    :param rotation_axis: 3-element tuple. The rotation axis
    :param translation: <np.float: 3, 1>. Translation in x, y, z direction.
    :param rotation: float. rotation in degrees
    :return:
    """

    # get inverse transform
    rot_mat = box.rotation_matrix
    trans = box.center

    new_box = copy.deepcopy(box)
    new_pc = copy.deepcopy(in_box_pc)

    new_pc.translate(-trans)
    new_box.translate(-trans)
    new_pc.rotate(rot_mat.T)
    new_box.rotate(Quaternion(matrix=rot_mat.T))

    if flip_x:
        new_pc.points[0, :] = -new_pc.points[0, :]
        # rotate the box to make sure that the x-axis is point to the head
        new_box.rotate(Quaternion(axis=[0, 0, 1], degrees=180))
    if flip_y:
        new_pc.points[1, :] = -new_pc.points[1, :]

    # apply rotation
    rot_quat = Quaternion(axis=rotation_axis, degrees=rotation)
    new_box.rotate(rot_quat)
    new_pc.rotate(rot_quat.rotation_matrix)

    # apply translation
    new_box.translate(translation)
    new_pc.translate(translation)

    # transform back
    new_box.rotate(Quaternion(matrix=rot_mat))
    new_pc.rotate(rot_mat)
    new_box.translate(trans)
    new_pc.translate(trans)
    return new_pc, new_box


def apply_augmentation(pc, box, wlh_factor=1.25):
    in_box_mask = nuscenes.utils.geometry_utils.points_in_box(box, pc.points, wlh_factor=wlh_factor)
    # in_box_mask = get_in_box_mask(pc, box)
    in_box_pc = PointCloud(pc.points[:, in_box_mask])

    rand_trans = np.random.uniform(low=-0.3, high=0.3, size=3)
    rand_rot = np.random.uniform(low=-10, high=10)
    flip_x, flip_y = np.random.choice([True, False], size=2, replace=True)

    new_in_box_pc, new_box = apply_transform(in_box_pc, box, rand_trans, rand_rot, flip_x, flip_y)

    new_pc = copy.deepcopy(pc)
    new_pc.points[:, in_box_mask] = new_in_box_pc.points
    return new_pc, new_box


def roty_batch_tensor(t):
    input_shape = t.shape
    output = torch.zeros(tuple(list(input_shape) + [3, 3]), dtype=torch.float32, device=t.device)
    c = torch.cos(t)
    s = torch.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output


def rotz_batch_tensor(t):
    input_shape = t.shape
    output = torch.zeros(tuple(list(input_shape) + [3, 3]), dtype=torch.float32, device=t.device)
    c = torch.cos(t)
    s = torch.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 1] = -s
    output[..., 1, 0] = s
    output[..., 1, 1] = c
    output[..., 2, 2] = 1
    return output


def get_offset_points_tensor(points, ref_box_params, offset_box_params):
    """

    :param points: B,N,3
    :param ref_box_params: B,4
    :return:
    """
    ref_center = ref_box_params[:, :3]
    ref_rot_angles = ref_box_params[:, -1]
    offset_center = offset_box_params[:, :3]
    offset_rot_angles = offset_box_params[:, -1]

    # transform to object coordinate system defined by the ref_box_params
    rot_mat = rotz_batch_tensor(-ref_rot_angles)  # B,3,3
    points -= ref_center[:, None, :]  # B,N,3
    points = torch.matmul(points, rot_mat.transpose(1, 2))

    # apply the offset
    rot_mat_offset = rotz_batch_tensor(offset_rot_angles)
    points = torch.matmul(points, rot_mat_offset.transpose(1, 2))
    points += offset_center[:, None, :]

    # # transform back to world coordinate
    points = torch.matmul(points, rot_mat)
    points += ref_center[:, None, :]
    return points


def get_offset_box_tensor(ref_box_params, offset_box_params):
    """
    transform the ref_box with the give offset
    :param ref_box_params: B,4
    :param offset_box_params: B,4
    :return: B,4
    """
    ref_center = ref_box_params[:, :3]  # B,3
    ref_rot_angles = ref_box_params[:, -1]  # B,
    offset_center = offset_box_params[:, :3]
    offset_rot_angles = offset_box_params[:, -1]
    rot_mat = rotz_batch_tensor(ref_rot_angles)  # B,3,3

    new_center = torch.matmul(rot_mat, offset_center[..., None]).squeeze(dim=-1)  # B,3
    new_center += ref_center
    new_angle = ref_rot_angles + offset_rot_angles
    return torch.cat([new_center, new_angle[:, None]], dim=-1)


def remove_transform_points_tensor(points, ref_box_params):
    """

    :param points: B,N,3
    :param ref_box_params: B,4
    :return:
    """
    ref_center = ref_box_params[:, :3]
    ref_rot_angles = ref_box_params[:, -1]

    # transform to object coordinate system defined by the ref_box_params
    rot_mat = rotz_batch_tensor(-ref_rot_angles)  # B,3,3
    points -= ref_center[:, None, :]  # B,N,3
    points = torch.matmul(points, rot_mat.transpose(1, 2))
    return points


def np_to_torch_tensor(data, device=None):
    return torch.tensor(data, device=device).unsqueeze(dim=0)


# for augmentation : post processing
def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[1] = -gt_boxes[1]

        if len(gt_boxes) == 4:
            gt_boxes[3] = -gt_boxes[3]
        else:
            gt_boxes[6] = -gt_boxes[6]

        points[:, 1] = -points[:, 1]

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points):
    """
    Args:
        gt_boxes: (7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[0] = -gt_boxes[0]

        if len(gt_boxes) == 4:
            gt_boxes[3] = -(gt_boxes[3] + np.pi)
        else:
            gt_boxes[6] = -(gt_boxes[6] + np.pi)
        points[:, 0] = -points[:, 0]

    return gt_boxes, points


def local_scaling(gt_boxes, points, noise_scale):
    """
    Args:
        gt_boxes: (7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """

    box = gt_boxes

    # noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    # augs[f'object_{idx}'] = noise_scale
    points_in_box, mask = get_points_in_box(points, box)

    # tranlation to axis center
    points[mask, 0] -= box[0]
    points[mask, 1] -= box[1]
    points[mask, 2] -= box[2]

    # apply scaling
    points[mask, :3] *= noise_scale

    # tranlation back to original position
    points[mask, 0] += box[0]
    points[mask, 1] += box[1]
    points[mask, 2] += box[2]

    gt_boxes[3:6] *= noise_scale
    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """

    gt_boxes = gt_boxes[np.newaxis, :]
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1], 1)
    points = rotate_points_along_z(points[np.newaxis, :, :], noise_rotation)[0]

    gt_boxes[:, 0:3] = rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], noise_rotation)[0]
    if gt_boxes.shape[1] == 4:
        gt_boxes[:, 4] += noise_rotation
    else:
        gt_boxes[:, 6] += noise_rotation

    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            noise_rotation
        )[0][:, 0:2]

    return gt_boxes[0], points


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    is_numpy = False
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float()
        is_numpy = True

    if isinstance(angle, np.ndarray):
        angle = torch.from_numpy(angle).float()

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros([points.shape[0]])
    ones = angle.new_ones([points.shape[0]])
    rot_matrix = torch.stack([
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ], dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def get_points_in_box(points, gt_box):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    cx, cy, cz = gt_box[0], gt_box[1], gt_box[2]
    dx, dy, dz, rz = gt_box[3], gt_box[4], gt_box[5], gt_box[6]
    shift_x, shift_y, shift_z = x - cx, y - cy, z - cz

    MARGIN = 1e-1
    cosa, sina = np.cos(-rz), np.sin(-rz)
    local_x = shift_x * cosa + shift_y * (-sina)
    local_y = shift_x * sina + shift_y * cosa

    mask = np.logical_and(abs(shift_z) <= dz / 2.0,
                          np.logical_and(abs(local_x) <= dx / 2.0 + MARGIN,
                                         abs(local_y) <= dy / 2.0 + MARGIN)
                          )

    points = points[mask]

    return points, mask


def lidar2bbox(boxs_para):
    # box: [[x, y, z, l, w, h, ry]] canonical coord, shape of [n,7]
    bboxs = []

    for b in boxs_para:
        center = [b[0], b[1], b[2]]
        size = [b[4], b[3], b[5]] #wlh
        orientation = Quaternion(axis=[0, 0, 1], radians=b[6])  # * Quaternion(axis=[1, 0, 0], radians=-np.pi/2)
        bbox = Box(center, size, orientation)
        # bbox.rotate(Quaternion(axis=[1, 0, 0], angle=np.pi/2))
        bboxs.append(bbox)

    return bboxs

