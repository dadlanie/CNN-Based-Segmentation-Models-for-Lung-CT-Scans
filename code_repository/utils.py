import sys
import random
import os
import torch
import numpy as np
import math



class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.2 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.25 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def iou_calculator(pred, label, n_class=3):
    mIoU = torch.zeros([n_class,1])

    for i in range(n_class):
        pred_pixel = pred==i
        label_pixel = label==i
        intersection = pred_pixel[label_pixel].sum()
        union = pred_pixel.sum()+label_pixel.sum()-intersection
        if union == 0:
            img_iou = math.nan
        else:
            img_iou = intersection/union
        mIoU[i] = img_iou

    return mIoU

