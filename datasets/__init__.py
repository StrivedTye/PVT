""" 
___init__.py
Created by zenn at 2021/7/18 15:50
"""
from datasets import kitti, sampler, nuscenes_data, waymo_data, pandaset_data


def get_dataset(config, type='train', **kwargs):
    if config.dataset == 'kitti':
        data = kitti.kittiDataset(path=config.path,
                                  split=kwargs.get('split', 'train'),
                                  category_name=kwargs.get('category_name', config.category_name),
                                  coordinate_mode=config.coordinate_mode,
                                  preloading=config.preloading,
                                  preload_offset=config.preload_offset if type != 'test' else -1)
    elif config.dataset == 'nuscenes':
        data = nuscenes_data.NuScenesDataset(path=config.path,
                                             split=kwargs.get('split', 'train_track'),
                                             category_name=kwargs.get('category_name', config.category_name),
                                             version=config.version,
                                             key_frame_only=True if type != 'test' else config.key_frame_only,
                                             preload_offset=config.preload_offset if type != 'test' else -1,
                                             preloading=config.preloading,
                                             min_points=config.min_points if kwargs.get('split', 'train_track') in
                                                             [config.val_split, config.test_split] else -1)
    elif config.dataset == 'waymo':
        data = waymo_data.WaymoDataset(path=config.path,
                                       split=kwargs.get('split', 'train'),
                                       category_name=kwargs.get('category_name', config.category_name),
                                       preload_offset=config.preload_offset,
                                       preloading=config.preloading,
                                       tiny=config.tiny)
    elif config.dataset == 'pandaset':
        data = pandaset_data.pandaDataset(path=config.path,
                                          split=kwargs.get('split', 'train'),
                                          category_name=kwargs.get('category_name', config.category_name),
                                          preload_offset=config.preload_offset if type != 'test' else -1,
                                          preloading=config.preloading,
                                          tiny=config.tiny)
    else:
        data = None

    if type == 'train_siamese':
        print('train number of frames: ', data.get_num_frames_total(), data.get_num_tracklets())
        return sampler.PointTrackingSampler(dataset=data,
                                            random_sample=config.random_sample,
                                            sample_per_epoch=config.sample_per_epoch,
                                            config=config)
    elif type.lower() == 'train_motion':
        return sampler.MotionTrackingSampler(dataset=data,
                                             config=config)
    else:
        print('test/val number of frames: ', data.get_num_frames_total(), data.get_num_tracklets())
        return sampler.TestTrackingSampler(dataset=data, config=config)
