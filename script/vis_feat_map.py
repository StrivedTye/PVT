"""
main.py
Created by zenn at 2021/7/18 15:08
"""
import pytorch_lightning as pl
import argparse

import pytorch_lightning.utilities.distributed
import torch
import yaml
from easydict import EasyDict
import os

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from datasets import get_dataset
from datasets.sampler import sparse_collate_fn
from models import get_model
from pytorch_lightning.trainer.supporters import CombinedLoader

# os.environ["NCCL_DEBUG"] = "INFO"


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--epoch', type=int, default=60, help='number of epochs')
    parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--gpu', type=int, nargs='+', default=(0, 1), help='specify gpu devices')
    parser.add_argument('--cfg', type=str, default='./cfgs/P2B.yaml', help='the config_file')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint location')
    parser.add_argument('--log_dir', type=str, default=None, help='log location')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--save_top_k', type=int, default=-1,
                        help='save top k checkpoints, use -1 to checkpoint every epoch')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='check_val_every_n_epoch')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload data into memory')
    parser.add_argument('--category_name', type=str, default='Car', help='use which category to train/test')
    parser.add_argument('--visual', action='store_true', default=False, help='visualization')
    # ablation study for pvt
    parser.add_argument('--base_scale', type=float, default=0.25, help='voxel size for feat fusion')
    parser.add_argument('--num_knn', type=int, default=12, help='knn for feat fusion')
    parser.add_argument('--backbone_voxel', type=float, default=12, help='knn for feat fusion')
    parser.add_argument('--re_weight', action='store_true', default=False, help='feature decorrelation')

    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))  # override the configuration using the value in args

    return EasyDict(config)


cfg = parse_config()

# init model
if cfg.checkpoint is None:
    net = get_model(cfg.net_model)(cfg)
else:
    net = get_model(cfg.net_model).load_from_checkpoint(cfg.checkpoint, config=cfg)

if not cfg.test:

    pin_mem = True

    if getattr(cfg, 'sparse_quantize', False):
        cfn = sparse_collate_fn
    else:
        cfn = None

    train_type = getattr(cfg, 'train_type', 'train_siamese')
    # dataset and dataloader
    if getattr(cfg, 'use_domains', False):
        train_loader = []
        for d in cfg.train_domains:
            train_data = get_dataset(cfg, type=train_type, split=cfg.train_split, category_name=d)
            temp_loader = DataLoader(train_data, batch_size=cfg.batch_size, num_workers=cfg.workers,
                                     shuffle=True, pin_memory=pin_mem, collate_fn=cfn)
            train_loader.append(temp_loader)

        val_data = get_dataset(cfg, type='test', split=cfg.val_split, category_name=cfg.test_domains)
        val_loader = DataLoader(val_data, batch_size=1, num_workers=cfg.workers,
                                collate_fn=lambda x: x, pin_memory=pin_mem)
    else:
        train_data = get_dataset(cfg, type=train_type, split=cfg.train_split)
        train_loader = DataLoader(train_data, batch_size=cfg.batch_size, num_workers=cfg.workers,
                                  shuffle=True, pin_memory=pin_mem, collate_fn=cfn)

        val_data = get_dataset(cfg, type='test', split=cfg.val_split)
        val_loader = DataLoader(val_data, batch_size=1, num_workers=cfg.workers,
                                collate_fn=lambda x: x, pin_memory=pin_mem)

    checkpoint_callback = ModelCheckpoint(monitor='precision/test', mode='max', save_last=True,
                                          save_top_k=cfg.save_top_k)

    # init trainer
    trainer = pl.Trainer(gpus=cfg.gpu, accelerator='ddp', max_epochs=cfg.epoch,
                         resume_from_checkpoint=cfg.checkpoint,
                         callbacks=[checkpoint_callback], default_root_dir=cfg.log_dir,
                         check_val_every_n_epoch=cfg.check_val_every_n_epoch, num_sanity_val_steps=2)
    trainer.fit(net, train_loader, val_loader)
    print(checkpoint_callback.best_model_path)
else:
    test_data = get_dataset(cfg, type='test', split=cfg.test_split)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=cfg.workers,
                             collate_fn=lambda x: x, pin_memory=True)

    trainer = pl.Trainer(gpus=cfg.gpu, accelerator='ddp', default_root_dir=cfg.log_dir,
                         resume_from_checkpoint=cfg.checkpoint)
    # trainer.test(net, test_loader)

    # visualize feature
    with torch.no_grad():
        for id, tracklet in enumerate(test_loader):
            if id == 5:
                sequcence = tracklet[0]
                ious, distances = net.cuda().evaluate_one_sequence(sequcence)

    # weight = net.rpn.FC_layer_cla[2].conv.weight # B, 1, N
    #
    # for n, p in net.named_parameters():
    #     print(n, p.size())
    # print(weight)

