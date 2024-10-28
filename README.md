# PVT
Pytorch-Lightning implementation of the PVT Tracker.  


### Setup
#### 1. DOCKER
+ Directly pull image form DockerHub
    ```
    docker pull strivedtye/mink:0.5.4-pytorch1.8.2-v2-sot3d
    ```
#### 2. Manually install by Conda
+ Create the environment
  ```
  git clone https://github.com/StrivedTye/PVT.git
  cd PVT
  conda create -n pvt  python=3.8
  conda activate pvt
  ```
+ Install pytorch
  ```
  conda install pytorch torchvision cudatoolkit=11.1 -c pytorch-lts -c nvidia
  ```
+ Install other dependencies:
  ```
  pip install -r requirement.txt
  ```
+ Install the nuscenes-devkit if you use want to use NuScenes dataset:
  ```
  pip install nuscenes-devkit
  ```
+ Install the torch-scatter
  ```
  conda install pytorch-scatter -c pyg
  ```
+ Install MinkowskiEngine
  ```
  please refer to [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
  ```
  

KITTI dataset
+ Download the data for [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).
+ Unzip the downloaded files.
+ Put the unzipped files under the same folder as following.
  ```
  [Parent Folder]
  --> [calib]
      --> {0000-0020}.txt
  --> [label_02]
      --> {0000-0020}.txt
  --> [velodyne]
      --> [0000-0020] folders with velodynes .bin files
  ```

NuScenes dataset
+ Download the dataset from the [download page](https://www.nuscenes.org/download)
+ Extract the downloaded files and make sure you have the following structure:
  ```
  [Parent Folder]
    samples	-	Sensor data for keyframes.
    sweeps	-	Sensor data for intermediate frames.
    maps	        -	Folder for all map files: rasterized .png images and vectorized .json files.
    v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
  ```
>Note: We use the **train_track** split to train our model and test it with the **val** split. Both splits are officially provided by NuScenes. During testing, we ignore the sequences where there is no point in the first given bbox.


### Quick Start
#### Training
To train a model, you must specify the `.yaml` file with `--cfg` argument. The `.yaml` file contains all the configurations of the dataset and the model. Currently, we provide four `.yaml` files under the [*cfgs*](./cfgs) directory. **Note:** Before running the code, you will need to edit the `.yaml` file by setting the `path` argument as the correct root of the dataset.
```bash
python main.py --gpu 0 1 --cfg cfgs/PVT_Car.yaml  --batch_size 50 --epoch 60
```
After you start training, you can start Tensorboard to monitor the training process:
```
tensorboard --logdir=./ --port=6006
```
By default, the trainer runs a full evaluation on the full test split after training every epoch. You can set `--check_val_every_n_epoch` to a larger number to speed up the training.
#### Testing
To test a trained model, specify the checkpoint location with `--checkpoint` argument and send the `--test` flag to the command.
```bash
python main.py --gpu 0 1 --cfg cfgs/PVT_Car.yaml  --checkpoint /path/to/checkpoint/xxx.ckpt --test
```


### Acknowledgment
+ This repo is built upon [Open3DSOT](https://github.com/Ghostish/Open3DSOT).
+ Thank Erik Wijmans for his pytorch implementation of [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch)
+ Thank the group of [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)