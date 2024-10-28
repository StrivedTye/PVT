from functools import partial
import numpy as np
import datasets.points_utils as augmentor_utils


class DataAugmentor(object):
    def __init__(self, config):
        self.config = config

    def random_world_flip(self, data_dict=None):

        gt_boxes, points = data_dict['box_search'], data_dict['search_points']
        gt_boxes2, points2 = data_dict['box_tmpl'], data_dict['template_points']

        for cur_axis in self.config['world_flip_axis']:
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points)
            gt_boxes2, points2 = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes2, points2)

        data_dict['box_search'] = gt_boxes
        data_dict['search_points'] = points
        data_dict['box_tmpl'] = gt_boxes2
        data_dict['template_points'] = points2
        return data_dict

    def random_world_rotation(self, data_dict=None):
        rot_range = self.config['world_rot_angle']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['box_search'], data_dict['search_points'], rot_range
        )

        data_dict['box_search'] = gt_boxes
        data_dict['search_points'] = points

        return data_dict

    def random_local_scaling(self, data_dict=None):
        """
        Please check the correctness of it before using.
        """
        scale_range = self.config['local_scale_range']
        if scale_range[1] - scale_range[0] < 1e-3:
            return data_dict
        noise_scale = np.random.uniform(scale_range[0], scale_range[1])

        gt_boxes, points = augmentor_utils.local_scaling(
            data_dict['box_search'], data_dict['search_points'], noise_scale)

        gt_boxes2, points2 = augmentor_utils.local_scaling(
            data_dict['box_tmpl'], data_dict['template_points'], noise_scale)

        data_dict['box_search'] = gt_boxes
        data_dict['search_points'] = points
        data_dict['box_tmpl'] = gt_boxes2
        data_dict['template_points'] = points2

        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...
        Returns:
        """
        data_dict = self.random_world_flip(data_dict)
        # data_dict = self.random_world_rotation(data_dict)
        data_dict = self.random_local_scaling(data_dict)

        return data_dict


if __name__ == "__main__":
    from datasets import kitti, sampler
    import yaml
    from easydict import EasyDict
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from datasets.data_classes import Box, PointCloud
    from datasets.points_utils import lidar2bbox

    file_name = "../cfgs/P2B_Car.yaml"
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    config = EasyDict(config)
    config.category_name = "Car"  #Car, Van, Pedestrian, Cyclist
    data = kitti.kittiDataset(path=config.path,
                              split=config.train_split,
                              category_name=config.category_name,
                              coordinate_mode=config.coordinate_mode,
                              preloading=False,
                              preload_offset=config.preload_offset if type != 'test' else -1)

    data_loader = sampler.PointTrackingSampler(dataset=data,
                                               random_sample=config.random_sample,
                                               sample_per_epoch=config.sample_per_epoch,
                                               config=config)

    data_dict = data_loader[100]

    view_PC = PointCloud(data_dict['search_points'].T)
    sample_semantic = data_dict['seg_label']
    gt_BB_ = lidar2bbox([data_dict['box_search']])[0]

    # Create figure for TRACKING
    fig = plt.figure(figsize=(15, 10), facecolor="white")
    plt.rcParams['savefig.dpi'] = 300
    # Create axis in 3D
    ax = fig.gca(projection='3d')

    # Scatter plot the cropped point cloud
    ratio = 1
    ax.scatter(
        view_PC.points[0, ::ratio],
        view_PC.points[1, ::ratio],
        view_PC.points[2, ::ratio],
        s=3,
        c=view_PC.points[0, ::ratio])

    flag = np.reshape(sample_semantic, [1, -1]).repeat(3, 0)
    pc = np.where(flag == 1., view_PC.points, 0)
    ax.scatter(
        pc[0, ::ratio],
        pc[1, ::ratio],
        pc[2, ::ratio],
        s=20, color='blue')

    # point order to draw a full Box
    order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]

    # # Plot Box
    # ax.plot(
    #     gt_BB_.corners()[0, order],
    #     gt_BB_.corners()[1, order],
    #     gt_BB_.corners()[2, order],
    #     color="red",
    #     alpha=0.5,
    #     linewidth=2,
    #     linestyle="-")

    ax.set_axis_off()
    ax.view_init(30, 120)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()



