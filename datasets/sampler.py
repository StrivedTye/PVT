# Created by zenn at 2021/4/27

import torch
import numpy as np
from easydict import EasyDict
from torch.utils.data._utils.collate import default_collate
from nuscenes.utils import geometry_utils
try:
    import MinkowskiEngine as ME
except:
    print("MinkowskiEngine is not installed.")

import datasets.points_utils as points_utils
from datasets.searchspace import KalmanFiltering
from datasets.data_augmentor import DataAugmentor


def no_processing(data, *args):
    return data


def siamese_processing(data, config, template_transform=None, search_transform=None):
    """

    :param data:
    :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
    :return:
    """
    first_frame = data['first_frame']
    template_frame = data['template_frame']
    search_frame = data['search_frame']
    candidate_id = data['candidate_id']
    first_pc, first_box = first_frame['pc'], first_frame['3d_bbox']
    template_pc, template_box = template_frame['pc'], template_frame['3d_bbox']
    search_pc, search_box = search_frame['pc'], search_frame['3d_bbox']

    use_z_offset = getattr(config, 'use_z_offset', False)
    limit_box_abs = getattr(config, 'limit_box_abs', False)  # only for training data

    if template_transform is not None:
        template_pc, template_box = template_transform(template_pc, template_box)
        first_pc, first_box = template_transform(first_pc, first_box)
    if search_transform is not None:
        search_pc, search_box = search_transform(search_pc, search_box)

    # generating template. Merging the object from previous and the first frames.
    if candidate_id == 0:
        samplegt_offsets = np.zeros(4)
    else:
        samplegt_offsets = np.random.uniform(low=-0.3, high=0.3, size=4)
        samplegt_offsets[3] = samplegt_offsets[3] * (5 if config.degrees else np.deg2rad(5))
    samplegt_offsets = samplegt_offsets if use_z_offset else samplegt_offsets[[0, 1, 3]]

    template_box = points_utils.getOffsetBB(template_box, samplegt_offsets,
                                            limit_box=config.limit_box, degrees=config.degrees,
                                            limit_box_abs=limit_box_abs, use_z=config.use_z)
    model_pc, model_box = points_utils.getModel([first_pc, template_pc], [first_box, template_box],
                                                scale=config.model_bb_scale, offset=config.model_bb_offset)

    assert model_pc.nbr_points() > 20, 'not enough template points'

    # generating search area. Use the current gt box to select the nearby region as the search area.
    if candidate_id == 0:
        # sample_offset = np.zeros(4)
        gaussian = KalmanFiltering(bnd=[0.1, 0.1, 0.05, (0.5 if config.degrees else np.deg2rad(0.5))]) # for stnet
    else:
        gaussian = KalmanFiltering(bnd=[1, 1, 0.5, (5 if config.degrees else np.deg2rad(5))])
    sample_offset = gaussian.sample(1)[0]  # for stnet
    sample_offset = sample_offset if use_z_offset else sample_offset[[0, 1, 3]]

    area_extents = getattr(config, 'area_extents', None)
    if area_extents is not None:
        area_extents = np.array(area_extents).reshape(3, 2)

    sample_bb = points_utils.getOffsetBB(search_box, sample_offset,
                                         limit_box=config.limit_box, degrees=config.degrees,
                                         limit_box_abs=limit_box_abs, use_z=config.use_z)
    search_pc_crop = points_utils.generate_subwindow(search_pc, sample_bb,
                                                     scale=config.search_bb_scale,
                                                     offset=config.search_bb_offset,
                                                     limit_area=area_extents)
    assert search_pc_crop.nbr_points() > 20, 'not enough search points'
    search_box = points_utils.transform_box(search_box, sample_bb)
    seg_label = points_utils.get_in_box_mask(search_pc_crop, search_box).astype(int)
    search_bbox_reg = np.array([search_box.center[0], search_box.center[1], search_box.center[2], -sample_offset[2]])

    template_points, idx_t = points_utils.regularize_pc(model_pc.points.T, config.template_size)
    search_points, idx_s = points_utils.regularize_pc(search_pc_crop.points.T, config.search_size)
    seg_label = seg_label[idx_s]

    data_dict = {
        'template_points': template_points.astype('float32'),
        'search_points': search_points.astype('float32'),
        'box_label': search_bbox_reg.astype('float32'),
        'bbox_size': search_box.wlh.astype('float32'),
        'seg_label': seg_label.astype('float32'),
    }

    if getattr(config, 'box_seven_param', False):
        box_tmpl = [0, 0, 0, model_box.wlh[1],  model_box.wlh[0],  model_box.wlh[2], 0]
        box_search = [search_box.center[0], search_box.center[1], search_box.center[2],
                      search_box.wlh[1], search_box.wlh[0], search_box.wlh[2],
                      np.deg2rad(search_bbox_reg[-1]) if config.degrees else search_bbox_reg[-1]
                      ]
        data_dict.update({'box_tmpl': np.array(box_tmpl).astype('float32'),
                          'box_search': np.array(box_search).astype('float32')})

    # augmentation
    if getattr(config, 'use_augment', False):
        aug = DataAugmentor(config)
        data_dict = aug.forward(data_dict)
        data_dict.update({'box_label': data_dict['box_search'].copy()[[0, 1, 2, 6]],
                          'bbox_size': data_dict['box_search'][[4, 3, 5]]})
        if config.degrees:
            data_dict['box_label'][-1] = np.rad2deg(data_dict['box_search'][-1])

    if getattr(config, 'box_aware', False):
        template_bc = points_utils.get_point_to_box_distance(template_points, model_box)
        search_bc = points_utils.get_point_to_box_distance(search_points, search_box)
        data_dict.update({'points2cc_dist_t': template_bc.astype('float32'),
                          'points2cc_dist_s': search_bc.astype('float32'), })

    if getattr(config, 'sparse_quantize', False):
        coords, inds = ME.utils.sparse_quantize(search_points,
                                                return_index=True,
                                                quantization_size=config.voxel_size)
        feats = search_points[inds]

        coords_t, inds_t = ME.utils.sparse_quantize(template_points,
                                                    return_index=True,
                                                    quantization_size=config.voxel_size)
        feats_t = template_points[inds_t]

        data_dict.update({'search_voxel': (coords, feats, inds),
                          'template_voxel': (coords_t, feats_t, inds_t)})

    if getattr(config, 'parallel_seeds', False):
        num_seed = 3
        seeds_offset = np.random.uniform(low=-0.3, high=0.3, size=[num_seed, 4])
        seeds_offset[:, -2] = seeds_offset[:, -2] * 0.5

        parallel_seeds = np.zeros([num_seed+1, 7])
        parallel_seeds[:, [3, 4, 5]] = search_box.wlh
        parallel_seeds[:, [0, 1, 2, 6]] = search_bbox_reg
        parallel_seeds[1:, [0, 1, 2, 6]] += seeds_offset
        data_dict.update({'parallel_seeds': parallel_seeds.astype('float32')})

    if getattr(config, 'use_voxel_rpn', False):
        voxel_size = np.array(config.voxel_size)
        xy_voxle_size = np.array(config.xy_size) * config.downsample

        area_extents = np.array(config.area_extents).reshape(3, 2)
        xy_area_extents = np.array(config.xy_area_extents).reshape(2, 2)
        voxel_extents_transpose = area_extents.transpose()
        extents_transpose = xy_area_extents.transpose()

        scene_ground = voxel_extents_transpose[0]
        voxel_grid_size = np.ceil(voxel_extents_transpose[1] / voxel_size) - np.floor(voxel_extents_transpose[0] / voxel_size)
        voxel_grid_size = voxel_grid_size.astype(np.int32)

        min_img_coord = np.floor(extents_transpose[0] / xy_voxle_size)
        max_img_coord = np.ceil(extents_transpose[1] / xy_voxle_size) - 1
        img_size = ((max_img_coord - min_img_coord) + 1).astype(np.int32)  # [w, h]

        offcenter = search_bbox_reg
        corners = search_box.corners()[:2].T
        corners = corners[[0, 2, 4, 6]]

        output_h = img_size[1]
        output_w = img_size[0]

        # hot_map
        corners_int = np.floor((corners / xy_voxle_size) - min_img_coord).astype(np.int32)
        corners_int_ul = np.min(corners_int, axis=0)
        corners_int_br = np.max(corners_int, axis=0)
        x = np.arange(corners_int_ul[0], corners_int_br[0]+1, 1)
        y = np.arange(corners_int_ul[1], corners_int_br[1]+1, 1)
        xx, yy = np.meshgrid(x, y)

        hot_map_grid = np.concatenate([xx[:, :, np.newaxis], yy[:, :, np.newaxis]], axis=2)
        ct = offcenter[:2]
        ct_image = (ct / xy_voxle_size) - min_img_coord
        ct_image_int = np.floor(ct_image).astype(np.int32)

        # (local_h, local_w)
        hot_map_grid = np.sqrt(np.sum((hot_map_grid-ct_image_int)**2, axis=2))
        hot_map_grid[hot_map_grid == 0] = 1e-6
        hot_map_grid = 1.0 / (hot_map_grid + 1e-6)
        # (1, h, w)
        hot_map = np.zeros((1, output_h, output_w), dtype=np.float32)
        # center: 1.0   around: 0.8     else: 0.0
        assert hot_map[0, corners_int_ul[1]:corners_int_br[1]+1, corners_int_ul[0]:corners_int_br[0]+1].shape == \
                (corners_int_br[1]+1-corners_int_ul[1], corners_int_br[0]+1-corners_int_ul[0]), 'not match shape of hot map'

        hot_map[0, corners_int_ul[1]:corners_int_br[1]+1, corners_int_ul[0]:corners_int_br[0]+1] = hot_map_grid
        hot_map[0, ct_image_int[1], ct_image_int[0]] = 1.0
        hot_map[0, [ct_image_int[1], ct_image_int[1], ct_image_int[1]+1, ct_image_int[1]-1], \
                [ct_image_int[0]-1, ct_image_int[0]+1, ct_image_int[0], ct_image_int[0]]] = 0.8

        # ((2r+1)^2,3) x,y,ry
        local_offsets = np.zeros(((2*config.regress_radius+1)**2, 3), dtype=np.float32)
        # (1,1)
        z_axis = np.array([[offcenter[2]]], dtype=np.float32)
        # center index
        index_center = np.array([ct_image_int[1]*output_w + ct_image_int[0]], dtype=np.int64)
        index_offsets = np.zeros(((2*config.regress_radius+1)**2,), dtype=np.int64)
        for i in range(-config.regress_radius, config.regress_radius+1):
            for j in range(-config.regress_radius, config.regress_radius+1):
                offsets = np.zeros((3,), dtype=np.float32)
                offsets[:2] = ct_image - (ct_image_int + np.array([i, j]))
                # rotate
                offsets[2] = offcenter[3]
                local_offsets[(i+config.regress_radius)*(2*config.regress_radius+1)+(j+config.regress_radius)] = offsets
                ind_int = ct_image_int + np.array([i, j])
                index_offsets[(i+config.regress_radius)*(2*config.regress_radius+1)+(j+config.regress_radius)] = ind_int[1]*output_w + ind_int[0]

        data_dict.update({'hot_map': hot_map,               # (1, H, W)
                          'index_center': index_center,     # (1, )
                          'z_axis': z_axis,                 # (1, 1)
                          'index_offsets': index_offsets,   # ((2*degress_ratio+1)**2,  ) ==> (25, )
                          'local_offsets': local_offsets})  # ((2*degress_ratio+1)**2, 3) ==> (25, 3)

    return data_dict


def motion_processing(data, config, template_transform=None, search_transform=None):
    """

    :param data:
    :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
    :return:
    point_sample_size
    bb_scale
    bb_offset
    """
    prev_frame = data['prev_frame']
    this_frame = data['this_frame']
    candidate_id = data['candidate_id']
    prev_pc, prev_box = prev_frame['pc'], prev_frame['3d_bbox']
    this_pc, this_box = this_frame['pc'], this_frame['3d_bbox']

    num_points_in_prev_box = geometry_utils.points_in_box(prev_box, prev_pc.points).sum()
    assert num_points_in_prev_box > 10, 'not enough target points'

    if template_transform is not None:
        prev_pc, prev_box = template_transform(prev_pc, prev_box)
    if search_transform is not None:
        this_pc, this_box = search_transform(this_pc, this_box)

    if candidate_id == 0:
        sample_offsets = np.zeros(3)
    else:
        sample_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
        sample_offsets[2] = sample_offsets[2] * (5 if config.degrees else np.deg2rad(5))
    ref_box = points_utils.getOffsetBB(prev_box, sample_offsets, limit_box=config.data_limit_box,
                                       degrees=config.degrees)
    prev_frame_pc = points_utils.generate_subwindow(prev_pc, ref_box,
                                                    scale=config.bb_scale,
                                                    offset=config.bb_offset)

    this_frame_pc = points_utils.generate_subwindow(this_pc, ref_box,
                                                    scale=config.bb_scale,
                                                    offset=config.bb_offset)
    assert this_frame_pc.nbr_points() > 20, 'not enough search points'

    this_box = points_utils.transform_box(this_box, ref_box)
    prev_box = points_utils.transform_box(prev_box, ref_box)
    ref_box = points_utils.transform_box(ref_box, ref_box)
    motion_box = points_utils.transform_box(this_box, prev_box)

    prev_points, idx_prev = points_utils.regularize_pc(prev_frame_pc.points.T, config.point_sample_size)
    this_points, idx_this = points_utils.regularize_pc(this_frame_pc.points.T, config.point_sample_size)

    seg_label_this = geometry_utils.points_in_box(this_box, this_points.T, 1.25).astype(int)
    seg_label_prev = geometry_utils.points_in_box(prev_box, prev_points.T, 1.25).astype(int)
    seg_mask_prev = geometry_utils.points_in_box(ref_box, prev_points.T, 1.25).astype(float)
    if candidate_id != 0:
        # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
        # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
        seg_mask_prev[seg_mask_prev == 0] = 0.2
        seg_mask_prev[seg_mask_prev == 1] = 0.8
    seg_mask_this = np.full(seg_mask_prev.shape, fill_value=0.5)

    timestamp_prev = np.full((config.point_sample_size, 1), fill_value=0)
    timestamp_this = np.full((config.point_sample_size, 1), fill_value=0.1)

    prev_points = np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]], axis=-1)
    this_points = np.concatenate([this_points, timestamp_this, seg_mask_this[:, None]], axis=-1)

    stack_points = np.concatenate([prev_points, this_points], axis=0)
    stack_seg_label = np.hstack([seg_label_prev, seg_label_this])
    theta_this = this_box.orientation.degrees * this_box.orientation.axis[-1] if config.degrees else \
        this_box.orientation.radians * this_box.orientation.axis[-1]
    box_label = np.append(this_box.center, theta_this).astype('float32')
    theta_prev = prev_box.orientation.degrees * prev_box.orientation.axis[-1] if config.degrees else \
        prev_box.orientation.radians * prev_box.orientation.axis[-1]
    box_label_prev = np.append(prev_box.center, theta_prev).astype('float32')
    theta_motion = motion_box.orientation.degrees * motion_box.orientation.axis[-1] if config.degrees else \
        motion_box.orientation.radians * motion_box.orientation.axis[-1]
    motion_label = np.append(motion_box.center, theta_motion).astype('float32')

    motion_state_label = np.sqrt(np.sum((this_box.center - prev_box.center) ** 2)) > config.motion_threshold

    data_dict = {
        'points': stack_points.astype('float32'),
        'box_label': box_label,
        'box_label_prev': box_label_prev,
        'motion_label': motion_label,
        'motion_state_label': motion_state_label.astype('int'),
        'bbox_size': this_box.wlh,
        'seg_label': stack_seg_label.astype('int'),
    }

    if getattr(config, 'box_aware', False):
        prev_bc = points_utils.get_point_to_box_distance(stack_points[:config.point_sample_size, :3], prev_box)
        this_bc = points_utils.get_point_to_box_distance(stack_points[config.point_sample_size:, :3], this_box)
        candidate_bc_prev = points_utils.get_point_to_box_distance(stack_points[:config.point_sample_size, :3], ref_box)
        candidate_bc_this = np.zeros_like(candidate_bc_prev)
        candidate_bc = np.concatenate([candidate_bc_prev, candidate_bc_this], axis=0)

        data_dict.update({'prev_bc': prev_bc.astype('float32'),
                          'this_bc': this_bc.astype('float32'),
                          'candidate_bc': candidate_bc.astype('float32')})
    return data_dict


def sparse_collate_fn(samples):
    data, search_voxel, template_voxel = [], [], []
    for sample in samples:
        search_voxel.append(sample['search_voxel'])
        template_voxel.append(sample['template_voxel'])
        data.append({w: sample[w] for w in sample if w != 'search_voxel' and w != 'template_voxel'})

    # for non-voxel data, use default collate
    data_batch = default_collate(data)

    # for search point cloud
    coords = [v[0] for v in search_voxel]
    feats = [v[1] for v in search_voxel]
    coords, feats = ME.utils.sparse_collate(coords, feats)
    inds = torch.cat([v[-1] for v in search_voxel], 0)
    data_batch['s_voxel_coords'] = coords
    data_batch['s_voxel_inds'] = inds
    data_batch['s_voxel_feats'] = feats

    # for template point cloud
    coords_t = [v[0] for v in template_voxel]
    feats_t = [v[1] for v in template_voxel]
    coords_t, feats_t = ME.utils.sparse_collate(coords_t, feats_t)
    inds_t = torch.cat([v[-1] for v in template_voxel], 0)
    data_batch['t_voxel_coords'] = coords_t
    data_batch['t_voxel_inds'] = inds_t
    data_batch['t_voxel_feats'] = feats_t

    return data_batch


class PointTrackingSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, random_sample, sample_per_epoch=10000, processing=siamese_processing, config=None,
                 **kwargs):
        if config is None:
            config = EasyDict(kwargs)
        self.sample_per_epoch = sample_per_epoch
        self.dataset = dataset
        self.processing = processing
        self.config = config
        self.random_sample = random_sample
        self.num_candidates = getattr(config, 'num_candidates', 1)

        if getattr(self.config, "use_augmentation", False):
            print('using augmentation')
            self.transform = points_utils.apply_augmentation
        else:
            self.transform = None

        if not self.random_sample:
            num_frames_total = 0
            self.tracklet_start_ids = [num_frames_total]
            for i in range(dataset.get_num_tracklets()):
                num_frames_total += dataset.get_num_frames_tracklet(i)
                self.tracklet_start_ids.append(num_frames_total)

    def get_anno_index(self, index):
        return index // self.num_candidates

    def get_candidate_index(self, index):
        return index % self.num_candidates

    def __len__(self):
        if self.random_sample:
            return self.sample_per_epoch * self.num_candidates
        else:
            return self.dataset.get_num_frames_total() * self.num_candidates

    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        try:
            if self.random_sample:
                tracklet_id = torch.randint(0, self.dataset.get_num_tracklets(), size=(1,)).item()
                tracklet_annos = self.dataset.tracklet_anno_list[tracklet_id]
                frame_ids = [0] + points_utils.random_choice(num_samples=2, size=len(tracklet_annos)).tolist()
            else:
                for i in range(0, self.dataset.get_num_tracklets()):
                    if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                        tracklet_id = i
                        this_frame_id = anno_id - self.tracklet_start_ids[i]
                        prev_frame_id = max(this_frame_id - 1, 0)
                        frame_ids = (0, prev_frame_id, this_frame_id)
            first_frame, template_frame, search_frame = self.dataset.get_frames(tracklet_id, frame_ids=frame_ids)
            data = {"first_frame": first_frame,
                    "template_frame": template_frame,
                    "search_frame": search_frame,
                    "candidate_id": candidate_id}

            return self.processing(data, self.config,
                                   template_transform=None,
                                   search_transform=self.transform)

        except AssertionError: # pvt
            return self[torch.randint(0, len(self), size=(1,)).item()]
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]
        except AssertionError:
            return self[0]


class TestTrackingSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, config=None, **kwargs):
        if config is None:
            config = EasyDict(kwargs)
        self.dataset = dataset
        self.config = config

    def __len__(self):
        return self.dataset.get_num_tracklets()

    def __getitem__(self, index):
        tracklet_annos = self.dataset.tracklet_anno_list[index]
        frame_ids = list(range(len(tracklet_annos)))
        return self.dataset.get_frames(index, frame_ids)


class MotionTrackingSampler(PointTrackingSampler):
    def __init__(self, dataset, config=None, **kwargs):
        super().__init__(dataset, random_sample=False, config=config, **kwargs)
        self.processing = motion_processing

    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        try:

            for i in range(0, self.dataset.get_num_tracklets()):
                if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                    tracklet_id = i
                    this_frame_id = anno_id - self.tracklet_start_ids[i]
                    prev_frame_id = max(this_frame_id - 1, 0)
                    frame_ids = (0, prev_frame_id, this_frame_id)
            first_frame, prev_frame, this_frame = self.dataset.get_frames(tracklet_id, frame_ids=frame_ids)
            data = {
                "first_frame": first_frame,
                "prev_frame": prev_frame,
                "this_frame": this_frame,
                "candidate_id": candidate_id}
            return self.processing(data, self.config,
                                   template_transform=self.transform,
                                   search_transform=self.transform)
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]
