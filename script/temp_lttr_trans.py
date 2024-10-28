import os
import numpy as np


if __name__ == "__main__1":
    root_dir = "./lttr_result_box/"
    classes = os.listdir(root_dir)
    for c in classes:
        path = os.path.join(root_dir, c)
        all_frame_txt = os.listdir(path)
        all_bb = []

        for frame in all_frame_txt:
            # print(os.path.join(path, frame))
            bb = np.loadtxt(os.path.join(path, frame)).T.flatten()
            all_bb.append(bb)

        track_ids = [int(f.split('_')[2]) for f in all_frame_txt]
        track_ids = np.array(track_ids)
        unique_track_ids = np.unique(track_ids)
        all_bb = np.array(all_bb)
        for track_id in unique_track_ids:
            cur_obj_bb = all_bb[track_ids == track_id]
            cur_obj_bb = np.vstack([cur_obj_bb[0:1, :], cur_obj_bb])
            print(os.path.join(root_dir, c, f'{track_id:04.0f}.txt'))
            np.savetxt(os.path.join(root_dir, c, f'{track_id:04.0f}.txt'), cur_obj_bb)
