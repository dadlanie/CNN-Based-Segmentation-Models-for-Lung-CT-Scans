import argparse
import time
import os
import sys
import random
import math

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from model import UNet, PSPNet, VGGNet, FCN8s
from utils import Logger, seed_everything, adjust_learning_rate, iou_calculator
from dataset import COVID19
from losses import FocalLoss2d



parser = argparse.ArgumentParser()

# experiment
parser.add_argument('--nettype',default='fcn',choices=['unet','pspnet','fcn'],help='choose from three candidate networks')
parser.add_argument('--loss',default='bce',choices=['bce','focal'])

# train
parser.add_argument('--epoch', type=int, default=40, help='the number of epochs being trained')
parser.add_argument('--batch_size', type=int, default=2, help='the size of a batch')
parser.add_argument('--numClasses', type=int, default=3, help='the number of classes')
parser.add_argument('--save_freq', type=int, default=10, help='save model/loss/acc every x epoch' )
parser.add_argument('--print_freq', type=int, default=10, help='print loss/acc every x iterations' )

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training')
parser.add_argument('--schedule',type=int,default=[], help='epochs to decrease learning rate')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--wd', type=float, default=1e-5, help='weight decay for training')

parser.add_argument('--SGDmomentum', type=float, default=0.9, help='momentum used for parameter update in SGD')

parser.add_argument('--save_path', type=str, default=r'C:\Users\WB\PycharmProjects\beng280a_pj1\checkpoints')

# device
parser.add_argument('--gpuid', type=int, default=0, help='GPU id to use.')
parser.add_argument('--workers', type=int, default=0, help='number of workers')

opt = parser.parse_args()

ROOT = r'C:\Users\WB\PycharmProjects\beng280a_pj1' # Change to your working folder

TRAIN_PATH = r'input\covid19-ct-scans\metadata_train.csv'
VAL_PATH = r'input\covid19-ct-scans\metadata_val.csv'


def train(train_loader,model,optimizer,criterion,device,epoch):
    adjust_learning_rate(optimizer, epoch, opt)
    lossmeter = []
    accmeter = []
    ioumeter_bg = []
    ioumeter_lung = []
    ioumeter_infect = []
    model.train().to(device)

    for i, (img, label_oh) in enumerate(train_loader):
        B = img.size(0)
        img = img.to(device)
        label = torch.argmax(label_oh, 1).to(device)
        label_oh = label_oh.to(device)

        # forward
        pred = model(img)
        prediction = torch.argmax(pred, dim=1)

        # loss
        if opt.loss == 'focal':
            loss = criterion(pred, label)
        else:
            if opt.nettype == 'fcn':
                loss = criterion(pred, label_oh)
            else:
                loss = criterion(pred, label)
        lossmeter.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for b in range(B):
            mIoU = iou_calculator(prediction[b, :, :], label[b, :, :], opt.numClasses)
            ioumeter_bg.append(mIoU[0].item())
            ioumeter_lung.append(mIoU[1].item())
            ioumeter_infect.append(mIoU[2].item())

        correct = prediction.eq_(label).view(-1)
        accuracy = float(torch.sum(correct)/len(correct))
        accmeter.append(accuracy)

        if (i+1) % opt.print_freq == 0:
            loss_mean = np.mean(lossmeter)
            acc_mean = np.mean(accmeter)
            print('Epoch[%d][%d/%d]--loss:%.4f(%.4f) acc:%.4f(%.4f) mIoU(infectious area):%.4f' % (epoch+1,i,
                                len(train_loader),loss,loss_mean,accuracy,acc_mean,np.nanmean(ioumeter_infect)))

    loss_mean = np.mean(lossmeter)
    mIoU_infect = np.nanmean(ioumeter_infect)
    # mIoU_lung = np.nanmean(ioumeter_lung)

    torch.cuda.empty_cache()

    return loss_mean, mIoU_infect



def validate(val_loader, model, criterion, device, epoch):
    lossmeter = []
    accmeter = []
    ioumeter_bg = []
    ioumeter_lung = []
    ioumeter_infect = []
    model.eval().to(device)

    with torch.no_grad():
        for i, (img, label_oh) in enumerate(val_loader):
            B = img.size(0)
            img = img.to(device)
            label = torch.argmax(label_oh, 1).to(device)
            label_oh = label_oh.to(device)

            # forward
            pred = model(img)
            prediction = torch.argmax(pred, dim=1)

            if opt.loss == 'focal':
                loss = criterion(pred, label)
            else:
                if opt.nettype == 'fcn':
                    loss = criterion(pred, label_oh)
                else:
                    loss = criterion(pred, label)
            lossmeter.append(loss.item())

            for b in range(B):
                mIoU = iou_calculator(prediction[b,:,:], label[b,:,:], opt.numClasses)
                ioumeter_bg.append(mIoU[0].item())
                ioumeter_lung.append(mIoU[1].item())
                ioumeter_infect.append(mIoU[2].item())

            correct = prediction.eq_(label).view(-1)
            accuracy = float(torch.sum(correct) / len(correct))
            accmeter.append(accuracy)

            if (i + 1) % opt.print_freq == 0:
                loss_mean = np.mean(lossmeter)
                acc_mean = np.mean(accmeter)
                print('Epoch[%d][%d/%d]--loss:%.4f(%.4f) acc:%.4f(%.4f) mIoU(infectious area):%.4f' % (epoch + 1, i,
                                           len(val_loader),loss, loss_mean,accuracy,acc_mean, np.nanmean(ioumeter_infect)))


    acc_mean = np.mean(accmeter)
    loss_mean = np.mean(lossmeter)
    mIoU_bg = np.nanmean(ioumeter_bg)
    mIoU_lung = np.nanmean(ioumeter_lung)
    mIoU_infect = np.nanmean(ioumeter_infect)
    if mIoU_infect == math.nan:
        mIoU_infect == 0

    print('Mean accuracy: %.4f' % (acc_mean))
    print('Mean IoU: background-%.4f; lung-%.4f; infectious area-%.4f; average-%.4f' % (mIoU_bg,
                                        mIoU_lung,mIoU_infect,(mIoU_bg+mIoU_lung+mIoU_infect)/3))

    torch.cuda.empty_cache()

    return loss_mean, mIoU_infect


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

    train_set = COVID19(os.path.join(ROOT, TRAIN_PATH), data_transform=True)
    val_set = COVID19(os.path.join(ROOT, VAL_PATH), data_transform=True)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    if opt.loss == 'focal':
        criterion = FocalLoss2d(1)
    else:
        if opt.nettype == 'fcn':
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

    lossmeter_train = []
    ioumeter_train = []
    lossmeter_val = []
    ioumeter_val = []
    best_infect_iou = 0

    for epoch in range(opt.epoch):
        print('---------------------Epoch[%d]---------------------' % (epoch+1))
        print('Training ... ')

        loss_train, iiou_train = train(train_loader,model,optimizer,criterion,device,epoch)
        lossmeter_train.append(loss_train)
        ioumeter_train.append(iiou_train)

        print('Validating ... ')
        loss_val, iiou_val = validate(val_loader, model, criterion, device, epoch)
        lossmeter_val.append(loss_val)
        ioumeter_val.append(iiou_val)

        if iiou_val > best_infect_iou:
            print('New Best mIoU of infected area (%.4f > %.4f)! Saving New Best Model ...' % (iiou_val, best_infect_iou))
            torch.save(model.state_dict(),
                       r'%s\best_model_%s@epoch_%d.pth' % (opt.save_path, opt.nettype, epoch))
            best_infect_iou = iiou_val

    # plot the training curve
    fig, ax1 = plt.subplots()
    ax1.plot(lossmeter_train,'b', label='train loss')
    ax1.plot(lossmeter_val,'r', label='validation loss')
    plt.legend()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    ax2 = ax1.twinx()
    ax2.plot(ioumeter_train, 'y', label='train mIoU')
    ax2.plot(ioumeter_val, 'g', label='validation mIoU')
    ax2.set_ylabel('mIoU of infected area')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    T = time.localtime()
    Time = '%d-%d-%d-%d-%d-%d' % (T[0],T[1],T[2],T[3],T[4],T[5])
    sys.stdout = Logger(r'logbooks/logbook_{}.txt'.format(Time), sys.stdout)
    main()