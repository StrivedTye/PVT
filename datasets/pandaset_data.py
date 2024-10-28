# Created by tye at 2021/10/

import copy
import random

from torch.utils.data import Dataset
from datasets.data_classes import PointCloud, Box
from pyquaternion import Quaternion
import numpy as np
import pandas as pd
import os
import warnings
import pickle
from collections import defaultdict
from datasets import points_utils, base_dataset
from pandaset import DataSet as pandaset



class pandaDataset(base_dataset.BaseDataset):
    def __init__(self, path, split, category_name="Car", **kwargs):
        super().__init__(path, split, category_name, **kwargs)
        self.coordinate_mode = kwargs.get('coordinate_mode', 'velodyne')
        self.preload_offset = kwargs.get('preload_offset', -1)
        self.tiny = kwargs.get('tiny', False)
        self.velos = defaultdict(dict)
        self.calibs = {}

        self.dataset = pandaset(path)
        self.scene_list = self._build_scene_list(split)
        self.tracklet_anno_list, self.tracklet_len_list = self._build_tracklet_anno()
        if self.preloading:
            self.training_samples = self._load_data()

    def _build_scene_list(self, split):
        if "TRAIN" in split.upper():  # Training SET
            if self.tiny:
                scene_names = [1]
            else:
                scene_names = list(range(1, 100))
        elif "VALID" in split.upper():  # Validation Set
            if self.tiny:
                scene_names = [80]
            else:
                scene_names = list(range(100, 110))
        elif "TEST" in split.upper():  # Testing Set
            if self.tiny:
                scene_names = [110]
            else:
                scene_names = list(range(110, 124))

        else:  # Full Dataset
            scene_names = list(range(124))
        scene_list = [scene for scene in self.dataset.sequences() if int(scene) in scene_names]

        return scene_list

    def _load_data(self):
        print('preloading data into memory')
        preload_data_path = os.path.join(self.dataset._directory,
                                         f"preload_panda_{self.category_name}_{self.split}_{self.preload_offset}.dat")
        if os.path.isfile(preload_data_path):
            print(f'loading from saved file {preload_data_path}.')
            with open(preload_data_path, 'rb') as f:
                training_samples = pickle.load(f)
        else:
            print('reading from annos')
            training_samples = []
            for i in range(len(self.tracklet_anno_list)):
                frames = []
                for anno in self.tracklet_anno_list[i]:
                    frames.append(self._get_frame_from_anno(anno))
                training_samples.append(frames)
            with open(preload_data_path, 'wb') as f:
                print(f'saving loaded data to {preload_data_path}')
                pickle.dump(training_samples, f)
        return training_samples

    def get_num_scenes(self):
        return len(self.scene_list)

    def get_num_tracklets(self):
        return len(self.tracklet_anno_list)

    def get_num_frames_total(self):
        return sum(self.tracklet_len_list)

    def get_num_frames_tracklet(self, tracklet_id):
        return self.tracklet_len_list[tracklet_id]

    def _build_tracklet_anno(self):
        list_of_tracklet_anno = []
        list_of_tracklet_len = []

        if "TRAIN" in self.split.upper():
            ratio = 5 # select one frame with the interval of 5
            for scene in self.scene_list:

                cur_scene = self.dataset[scene]
                cur_scene.load_cuboids()

                box_all = cur_scene.cuboids[::ratio]
                for frame, box in enumerate(box_all):
                    box.insert(loc=0, column="frame", value=frame*ratio)

                df = pd.concat(box_all, axis=0)

                if self.category_name == "All":
                    df0 = df[df["label"] == "Car"]
                    df2 = df[df["label"] == "Bus"]
                    df1 = df[df["label"] == "Pedestrian"]
                    df3 = df[df["label"] == "Bicycle"]
                    df = pd.concat([df2, df1, df3], axis=0)
                else:
                    df = df[df["label"] == self.category_name]

                df.insert(loc=0, column="scene", value=scene)
                df = df.rename(columns={'label': 'type'})

                list_track_id = sorted(df.uuid.unique())[::2] #[::8] #[::2]#
                for track_id in list_track_id:
                    df_tracklet = df[df["uuid"] == track_id]
                    df_tracklet = df_tracklet.reset_index(drop=True)
                    tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                    list_of_tracklet_anno.append(tracklet_anno)
                    list_of_tracklet_len.append((len(tracklet_anno)))
        else:
            for scene in self.scene_list:

                cur_scene = self.dataset[scene]
                cur_scene.load_cuboids()

                box_all = cur_scene.cuboids
                for frame, box in enumerate(box_all):
                    box.insert(loc=0, column="frame", value=frame)

                df = pd.concat(box_all, axis=0) # all frames

                if self.category_name == "All":
                    df0 = df[df["label"] == "Car"]
                    df1 = df[df["label"] == "Bus"]
                    df2 = df[df["label"] == "Bicycle"]

                    # only use moving targets except Pedestrian class
                    df3 = pd.concat([df2, df1], axis=0)
                    df3 = df3[df3['attributes.object_motion'] == 'Moving']

                    df4 = df[df["label"] == "Pedestrian"]
                    df = pd.concat([df3, df4], axis=0)
                else:
                    df = df[df["label"] == self.category_name]
                    df = df[df['attributes.object_motion'] == 'Moving']

                df.insert(loc=0, column="scene", value=scene)
                df = df.rename(columns={'label': 'type'})

                list_track_id = sorted(df.uuid.unique())#[::3]
                for track_id in list_track_id:
                    df_tracklet = df[df["uuid"] == track_id]
                    df_tracklet = df_tracklet.reset_index(drop=True)
                    tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                    list_of_tracklet_anno.append(tracklet_anno)
                    list_of_tracklet_len.append((len(tracklet_anno)))

        return list_of_tracklet_anno, list_of_tracklet_len

    def get_frames(self, seq_id, frame_ids):
        if self.preloading:
            frames = [self.training_samples[seq_id][f_id] for f_id in frame_ids]
        else:
            seq_annos = self.tracklet_anno_list[seq_id]
            frames = [self._get_frame_from_anno(seq_annos[f_id]) for f_id in frame_ids]

        return frames

    def _get_frame_from_anno(self, box):
        scene_id = box['scene']
        frame_id = box['frame']

        center = [box["position.x"], box["position.y"], box["position.z"]]
        size = [box["dimensions.x"], box["dimensions.y"], box["dimensions.z"]]
        orientation = Quaternion(axis=[0, 0, 1], radians=box["yaw"]) \
                      * Quaternion(axis=[0, 0, 1], radians=-np.pi / 2)
        BB = Box(center, size, orientation)

        try:
            try:
                PC = self.velos[scene_id][frame_id]
            except KeyError:
                # VELODYNE PointCloud
                fp = os.path.join(self.dataset._directory, box["scene"], 'lidar', '{:02}.pkl.gz'.format(box["frame"]))
                PC = pd.read_pickle(fp)
                PC = PC.values
                PC = PointCloud(PC[:, 0:3].T)
                # self.velos[scene_id][frame_id] = PC

            if self.preload_offset > 0:
                PC = points_utils.crop_pc_axis_aligned(PC, BB, offset=self.preload_offset)
        except :
            # in case the Point cloud is missing
            PC = PointCloud(np.array([[0, 0, 0]]).T)

        # todo add image
        return {"pc": PC, "3d_bbox": BB, 'meta': box}
