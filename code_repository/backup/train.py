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

from model import encoder_basic,decoder_basic,encoderDilation,decoderDilation,encoderSPP,decoderSPP
from utils import Logger, seed_everything, adjust_learning_rate, iou_calculator
from dataset import COVID19

'''
spp: lr = 5e-5  epoch = 60[3,40]
dilation: lr = 1e-4 epoch = 60[]

'''


parser = argparse.ArgumentParser()

# experiment
parser.add_argument('--nettype',default='basic',choices=['basic','dilation','spp'],help='choose from three candidate networks')

# train
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs being trained')
parser.add_argument('--batch_size', type=int, default=4, help='the size of a batch')
parser.add_argument('--numClasses', type=int, default=3, help='the number of classes')
parser.add_argument('--save_freq', type=int, default=10, help='save model/loss/acc every x epoch' )
parser.add_argument('--print_freq', type=int, default=10, help='print loss/acc every x iterations' )

parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for training')
parser.add_argument('--schedule',type=int,default=[10], help='epochs to decrease learning rate')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--wd', type=float, default=0, help='weight decay for training')

parser.add_argument('--SGDmomentum', type=float, default=0.9, help='momentum used for parameter update in SGD')

parser.add_argument('--save_path', type=str, default=r'C:\Users\WB\PycharmProjects\beng280a_pj1\checkpoints')

# device
parser.add_argument('--gpuid', type=int, default=0, help='GPU id to use.')
parser.add_argument('--workers', type=int, default=0, help='number of workers')

opt = parser.parse_args()

ROOT = r'C:\Users\WB\PycharmProjects\beng280a_pj1' # Change to your working folder

TRAIN_PATH = r'input\covid19-ct-scans\metadata_train.csv'
VAL_PATH = r'input\covid19-ct-scans\metadata_val.csv'


def train(train_loader,encoder,decoder,optimizer,criterion,device,epoch):
    adjust_learning_rate(optimizer, epoch, opt)
    lossmeter = []
    accmeter = []
    ioumeter_bg = []
    ioumeter_lung = []
    ioumeter_infect = []
    encoder.train().to(device)
    decoder.train().to(device)

    for i, (img, label_oh) in enumerate(train_loader):
        B = img.size(0)
        img = img.to(device)
        label = torch.argmax(label_oh, 1).to(device)
        label_oh = label_oh.to(device)


        # forward
        x1, x2, x3, x4, x5 = encoder(img)
        pred = decoder(img, x1, x2, x3, x4, x5)
        prediction = torch.argmax(pred, dim=1)

        # to_test = prediction.clone()
        # wrong = 1 - to_test.eq_(label)
        # infect_penalty = 10 * torch.mul(label_oh[:, 2, :, :],  wrong) + 1

        # loss
        loss = criterion(pred, label_oh)
        # loss[:,2,:,:] = torch.mul(loss[:,2,:,:], infect_penalty)
        # loss = torch.mean(loss)
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
    mIoU_lung = np.nanmean(ioumeter_lung)

    torch.cuda.empty_cache()

    return loss_mean, mIoU_lung



def validate(val_loader, encoder, decoder, criterion, device, epoch):
    lossmeter = []
    accmeter = []
    ioumeter_bg = []
    ioumeter_lung = []
    ioumeter_infect = []
    encoder.eval().to(device)
    decoder.eval().to(device)

    with torch.no_grad():
        for i, (img, label_oh) in enumerate(val_loader):
            B = img.size(0)
            img = img.to(device)
            label = torch.argmax(label_oh, 1).to(device)
            label_oh = label_oh.to(device)

            # forward
            x1, x2, x3, x4, x5 = encoder(img)
            pred = decoder(img, x1, x2, x3, x4, x5)

            # loss
            loss = criterion(pred, label_oh)
            # loss = torch.mean(loss)
            lossmeter.append(loss.item())

            prediction = torch.argmax(pred, dim=1)

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

    return loss_mean, mIoU_lung


def main():
    #seed_everything(random.randint(1, 10000))
    seed_everything(42)

    if torch.cuda.is_available() == True:
        device = torch.device('cuda:{}'.format(opt.gpuid))
        print('Use GPU for training')
    else:
        device = torch.device('cpu')
        print('Use CPU for training')

    # Initialize network
    if opt.nettype == 'dilation':
        encoder = encoderDilation()
        #model.loadPretrainedWeight(encoder)
        decoder = decoderDilation()
    elif opt.nettype == 'spp':
        encoder = encoderSPP()
        #model.loadPretrainedWeight(encoder)
        decoder = decoderSPP()
    else:
        encoder = encoder_basic()
        #model.loadPretrainedWeight(encoder)
        decoder = decoder_basic()

    train_set = COVID19(os.path.join(ROOT, TRAIN_PATH), data_transform=True)
    val_set = COVID19(os.path.join(ROOT, VAL_PATH), data_transform=True)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=opt.lr, weight_decay=opt.wd)
    #optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=opt.lr, momentum=opt.SGDmomentum, weight_decay=opt.wd)
    #criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCELoss(reduction='none')
    criterion = torch.nn.BCELoss()

    lossmeter_train = []
    ioumeter_train = []
    lossmeter_val = []
    ioumeter_val = []
    best_infect_iou = 0

    for epoch in range(opt.epoch):
        print('---------------------Epoch[%d]---------------------' % (epoch+1))
        print('Training ... ')

        loss_train, iiou_train = train(train_loader,encoder,decoder,optimizer,criterion,device,epoch)
        lossmeter_train.append(loss_train)
        ioumeter_train.append(iiou_train)

        print('Validating ... ')
        loss_val, iiou_val = validate(val_loader, encoder, decoder, criterion, device, epoch)
        lossmeter_val.append(loss_val)
        ioumeter_val.append(iiou_val)

        if iiou_val > best_infect_iou:
            print('New Best mIoU of lung area (%.4f > %.4f)! Saving New Best Model ...' % (iiou_val, best_infect_iou))
            torch.save(encoder.state_dict(),
                       r'%s\best_model(encoder)_%s@epoch_%d.pth' % (opt.save_path, opt.nettype, epoch))
            torch.save(decoder.state_dict(),
                       r'%s\best_model(decoder)_%s@epoch_%d.pth' % (opt.save_path, opt.nettype, epoch))
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
    ax2.set_ylabel('mIoU of lung area')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    T = time.localtime()
    Time = '%d-%d-%d-%d-%d-%d' % (T[0],T[1],T[2],T[3],T[4],T[5])
    sys.stdout = Logger(r'logbooks/logbook_{}.txt'.format(Time), sys.stdout)
    main()