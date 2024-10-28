import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from utils.loss_reweighting import lossb_expect


def lr_setter(optimizer, epoch, args, bl=False):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = args.lrbl * (0.1 ** (epoch // (args.epochb * 0.5)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weight_learner(cfeatures, pre_features, pre_weight1, args, global_epoch=0, iter=0):
    softmax = nn.Softmax(0)
    weight = Variable(torch.ones(cfeatures.size(0), 1).cuda())
    weight.requires_grad = True

    cfeaturec = Variable(torch.FloatTensor(cfeatures.size()).cuda())
    cfeaturec.data.copy_(cfeatures.data)

    all_feature = torch.cat([cfeaturec, pre_features.detach()], dim=0)
    optimizerbl = torch.optim.SGD([weight], lr=args.lrbl, momentum=0.9)

    for epoch in range(args.epochb):
        lr_setter(optimizerbl, epoch, args, bl=True)
        all_weight = torch.cat((weight, pre_weight1.detach()), dim=0)
        optimizerbl.zero_grad()

        lossb = lossb_expect(all_feature, softmax(all_weight), args.num_f)
        lossp = softmax(weight).pow(args.decay_pow).sum()
        lambdap = args.lambdap * max((args.lambda_decay_rate ** (global_epoch // args.lambda_decay_epoch)),
                                     args.min_lambda_times)
        lossg = lossb / lambdap + lossp

        if global_epoch == 0:
            lossg = lossg * 1.0

        lossg.backward(retain_graph=True)
        optimizerbl.step()

    if global_epoch == 0 and iter < 10:
        pre_features = (pre_features * iter + cfeatures) / (iter + 1)
        pre_weight1 = (pre_weight1 * iter + weight) / (iter + 1)

    elif cfeatures.size(0) < pre_features.size(0):
        pre_features[:cfeatures.size(0)] = pre_features[:cfeatures.size(0)] * args.presave_ratio + cfeatures * (
                    1 - args.presave_ratio)
        pre_weight1[:cfeatures.size(0)] = pre_weight1[:cfeatures.size(0)] * args.presave_ratio + weight * (
                    1 - args.presave_ratio)

    else:
        pre_features = pre_features * args.presave_ratio + cfeatures * (1 - args.presave_ratio)
        pre_weight1 = pre_weight1 * args.presave_ratio + weight * (1 - args.presave_ratio)

    softmax_weight = softmax(weight)

    return softmax_weight, pre_features, pre_weight1