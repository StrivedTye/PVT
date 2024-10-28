import numpy as np
from datasets import kitti, sampler
import yaml
from easydict import EasyDict
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from datasets.data_classes import Box, PointCloud
from datasets.points_utils import get_in_box_mask, crop_pc_axis_aligned


if __name__ == "__main__":
    file_name = "./cfgs/BAT_Car.yaml"
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    config = EasyDict(config)
    config.category_name = "Car"  #Car, Van, Pedestrian, Cyclist

    data = kitti.kittiDataset(path=config.path,
                              split=config.test_split,
                              category_name=config.category_name,
                              coordinate_mode=config.coordinate_mode,
                              preloading=False,
                              preload_offset=config.preload_offset if type != 'test' else -1)

    data_loader = sampler.TestTrackingSampler(dataset=data, config=config)

    pred_bb_root_dir = '../BAT/visual_results/'

    trackers_name = ['BAT', 'PVT', 'LTTR']
    trackers_color = ['seagreen', 'blue', 'orange']

    save_image_dir = './visual_image'

    # Create figure for TRACKING
    fig = plt.figure(figsize=(9, 6), facecolor="white")
    plt.rcParams['savefig.dpi'] = 300

    # Create axis in 3D
    # ax = fig.add_axes(Axes3D(fig))
    ax = fig.add_subplot(111, projection='3d')

    # create dir for saving images
    os.makedirs(os.path.join(save_image_dir, 'kitti', config.category_name), exist_ok=True)

    plt.ion() # start interaction plot
    for tracklet_id in range(len(data_loader)):
        instance = data_loader[tracklet_id]  # {"pc": pc, "3d_bbox": bb, 'meta': anno}

        # load trackers' prediction
        pred_bb = {}
        for tracker in trackers_name:
            pred_bb_path = os.path.join(pred_bb_root_dir, 'kitti_'+tracker,
                                        config.category_name, f'{tracklet_id:04.0f}.txt')
            pred_bb.update({tracker: np.loadtxt(pred_bb_path)})

        for frame_id in range(len(instance)):
            plt.cla()  # clear previous frame
            search_pc = instance[frame_id]['pc']
            gt_bb = instance[frame_id]['3d_bbox']
            search_pc = crop_pc_axis_aligned(search_pc, gt_bb, 3)

            # Scatter plot the cropped point cloud
            ratio = 1
            ax.scatter(search_pc.points[0, ::ratio],
                       search_pc.points[1, ::ratio],
                       search_pc.points[2, ::ratio],
                       s=3, color='gray')  # plot all points

            mask = get_in_box_mask(search_pc, gt_bb)
            flag = np.reshape(mask, [1, -1]).repeat(3, 0)
            pc = np.where(flag == 1., search_pc.points, 0)
            pc = pc[:, np.any(pc, 0)]

            ax.scatter(pc[0, ::ratio],
                       pc[1, ::ratio],
                       pc[2, ::ratio],
                       s=20, color='blue')  # highlight foreground points

            # point order to draw a full Box
            order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]

            # Plot Box
            ax.plot(gt_bb.corners()[0, order],
                    gt_bb.corners()[1, order],
                    gt_bb.corners()[2, order],
                    color="red",
                    alpha=0.5,
                    linewidth=2,
                    linestyle="-")

            for i, tracker in enumerate(trackers_name):
                try:
                    cur_bb = pred_bb[tracker][frame_id].reshape([3, 8])
                except ValueError:
                    cur_bb = pred_bb[tracker].reshape([3, 8])
                ax.plot(cur_bb[0, order],
                        cur_bb[1, order],
                        cur_bb[2, order],
                        color=trackers_color[i],
                        alpha=0.5,
                        linewidth=2,
                        linestyle="-")

            ax.set_axis_off()
            ax.view_init(30, 120)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.pause(0.001)
            print(os.path.join(save_image_dir, 'kitti', config.category_name, f'{tracklet_id:04.0f}_{frame_id:04.0f}.jpg'))
            # plt.savefig(os.path.join(save_image_dir, 'kitti', config.category_name, f'{tracklet_id:04.0f}_{frame_id:04.0f}.jpg'))

    plt.ioff()
    plt.show()
