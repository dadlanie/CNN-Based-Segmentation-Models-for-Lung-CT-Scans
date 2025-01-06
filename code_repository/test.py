import argparse
import time
import os
import sys
import random
import math

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from model import UNet, PSPNet, VGGNet, FCN8s
from utils import seed_everything, iou_calculator
from dataset import COVID19

parser = argparse.ArgumentParser()

# experiment
parser.add_argument('--nettype',default='fcn',choices=['unet','pspnet','fcn'],help='choose from three candidate networks')

# train
parser.add_argument('--batch_size', type=int, default=2, help='the size of a batch')
parser.add_argument('--numClasses', type=int, default=3, help='the number of classes')

parser.add_argument('--saved_model', type=str, default=r'C:\Users\WB\PycharmProjects\beng280a_pj1\checkpoints\ablation\best_model_fcn_bz2@epoch_7.pth')

# device
parser.add_argument('--gpuid', type=int, default=0, help='GPU id to use.')
parser.add_argument('--workers', type=int, default=0, help='number of workers')

opt = parser.parse_args()

ROOT = r"C:\Users\WB\PycharmProjects\beng280a_pj1" # Change to your working folder

TEST_PATH = r'input\covid19-ct-scans\metadata_test.csv'

def test(test_loader, model, device):
    accmeter = []
    ioumeter_bg = []
    ioumeter_lung = []
    ioumeter_infect = []
    model.eval().to(device)

    pred_record = torch.zeros([900, 512, 512])
    label_record = torch.zeros([900, 512, 512])
    img_record = torch.zeros([900, 512, 512])

    with torch.no_grad():
        for i, (img, label_oh) in enumerate(test_loader):
            B = img.size(0)
            img = img.to(device)
            label = torch.argmax(label_oh, 1).to(device)

            # forward
            pred = model(img)
            prediction = torch.argmax(pred, dim=1)

            for b in range(B):
                mIoU = iou_calculator(prediction[b,:,:], label[b,:,:], opt.numClasses)
                ioumeter_bg.append(mIoU[0].item())
                ioumeter_lung.append(mIoU[1].item())
                ioumeter_infect.append(mIoU[2].item())

                totest = prediction.clone()
                correct = totest.eq_(label).view(-1)
                accuracy = float(torch.sum(correct) / len(correct))
                accmeter.append(accuracy)

                pred_record[B*i+b,:,:] = prediction[b,:,:]
                img_record[B*i+b,:,:] = img[b,:,:]
                label_record[B*i+b,:,:] = label[b,:,:]


    print('Mean framewise accuracy: %.4f\n' % (np.mean(accmeter)))
    print('Mean IoU: background-%.4f; lung-%.4f; infection-%.4f\n' %
          (np.nanmean(ioumeter_bg),np.nanmean(ioumeter_lung),np.nanmean(ioumeter_infect)))

    plot_idx = 250

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle('Accuracy: %.4f; Lung mIoU: %.4f; Infection mIoU: %.4f' % (accmeter[plot_idx], ioumeter_lung[plot_idx], ioumeter_infect[plot_idx]))
    plt.subplot(1, 3, 1)
    plt.imshow(img_record[plot_idx,:,:], cmap='bone')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(img_record[plot_idx,:,:], cmap='bone')
    plt.imshow(label_record[plot_idx,:,:], alpha=0.5, cmap='nipy_spectral')
    plt.title('Ground Truth')

    plt.subplot(1, 3, 3)
    plt.imshow(img_record[plot_idx,:,:], cmap='bone')
    plt.imshow(pred_record[plot_idx,:,:], alpha=0.5, cmap='nipy_spectral')
    plt.title('Prediction')
    plt.show()




def main():
    seed_everything(random.randint(1, 10000))

    if torch.cuda.is_available() == True:
        device = torch.device('cuda:{}'.format(opt.gpuid))
        print('Use GPU for training')
    else:
        device = torch.device('cpu')
        print('Use CPU for training')

    # Initialize network
    if opt.nettype == 'unet':
        model = UNet(1, opt.numClasses)
    elif opt.nettype == 'pspnet':
        model = PSPNet(1, opt.numClasses)
    elif opt.nettype == 'fcn':
        vgg_model = VGGNet(requires_grad=True, pretrained=False)
        fcn_model = FCN8s(pretrained_net=vgg_model, n_class=opt.numClasses)
        model = fcn_model

    # loading trained model parameters
    if opt.saved_model:
        if os.path.isfile(opt.saved_model):
            print("=> loading pretrained model: '{}'".format(opt.saved_model))
            checkpoint = torch.load(opt.saved_model)
            model.load_state_dict(checkpoint, strict=False)
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(opt.saved_model))


    test_set = COVID19(os.path.join(ROOT, TEST_PATH), data_transform=True)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)

    test(test_loader, model, device)


if __name__ == '__main__':
    main()