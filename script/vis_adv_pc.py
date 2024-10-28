import numpy as np
import glob
import torch
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.graph_spectral import eig_vector

from hausdorff import hausdorff_distance
from chamferdist import ChamferDistance

chamferdist = ChamferDistance()

tracker_name = 'BAT'
category = 'Car' # Pedestrian, Van, Cyslist
attack_type = "cw"
ADV_DIR = '../adv_examples/' + tracker_name + '/' + attack_type + '_' + category + '/'

adv_sample_pathes = sorted(glob.glob(ADV_DIR+'adv_*.txt'))
clean_sample_pathes = sorted(glob.glob(ADV_DIR+'clean_*.txt'))

adv_pc = np.loadtxt(adv_sample_pathes[0])
clean_pc = np.loadtxt(clean_sample_pathes[0])

hd = hausdorff_distance(adv_pc, clean_pc, distance='euclidean')
cd = chamferdist(torch.tensor(clean_pc).unsqueeze(0).float(), torch.tensor(adv_pc).unsqueeze(0).float(), bidirectional=True)
print("HD:{:f}, CD:{:f}".format(hd, cd))

fig = plt.figure()
ax = fig.add_axes(Axes3D(fig))

ax.scatter(adv_pc[:, 0], adv_pc[:, 1], adv_pc[:, 2], color='blue')
ax.scatter(clean_pc[:, 0], clean_pc[:, 1], clean_pc[:, 2], color='green')

if adv_pc.shape[0] > 1024:
    ax.scatter(adv_pc[1024:, 0], adv_pc[1024:, 1], adv_pc[1024:, 2], color='red')
    ax.scatter(clean_pc[1024:, 0], clean_pc[1024:, 1], clean_pc[1024:, 2], color='cyan')


# # Graph spectral domain
# clean_pc = np.unique(clean_pc, axis=0)
# x = torch.from_numpy(clean_pc).unsqueeze(0).float()  # [1, n, 3]
# v, L, u = eig_vector(x, 10)  # v: eigien vector; u: eigien value
#
# u_sort, u_sort_ind = torch.sort(u, dim=1, descending=False)
# v = torch.gather(v, dim=2, index=u_sort_ind.unsqueeze(1).repeat(1, x.shape[1], 1))
#
# x_gft = torch.einsum('bij,bjk->bik', v.transpose(1, 2), x)  # (b,n,3)
#
# mask = torch.ones_like(x)
# mask[:, :100, :] = 1  # low
# mask[:, 100:400, :] = 0  # middle
# mask[:, 400:, :] = 0  # high
#
# # x_gft = x_gft * mask
#
# # IGFT
# x_hat = torch.einsum('bij,bjk->bik', v, x_gft)
#
# fig2 = plt.figure()
# ax2 = fig2.add_axes(Axes3D(fig2))
# ax2.scatter(x[0, :, 0], x[0, :, 1], x[0, :, 2], color='green')
# ax2.scatter(x_hat[0, :, 0], x_hat[0, :, 1], x_hat[0, :, 2], color='blue')
# ax2.set_axis_off()
#
# # plt.plot(np.arange(0, x_gft.shape[1]), x_gft[0, :, 0])
# # plt.plot(np.arange(0, x_gft.shape[1]), x_gft[0, :, 1])
# # plt.plot(np.arange(0, x_gft.shape[1]), x_gft[0, :, 2])

# ax.set_axis_off()
plt.show()
