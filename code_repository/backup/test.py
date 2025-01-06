import argparse
import time
import os
import sys
import random

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from model import encoder_basic,decoder_basic,encoderDilation,decoderDilation,encoderSPP,decoderSPP
from utils import Logger, seed_everything, adjust_learning_rate, iou_calculator
from dataset import COVID19

parser = argparse.ArgumentParser()

# experiment
parser.add_argument('--nettype',default='spp',choices=['basic','dilation','spp'],help='choose from three candidate networks')

# train
parser.add_argument('--batch_size', type=int, default=8, help='the size of a batch')
parser.add_argument('--numClasses', type=int, default=4, help='the number of classes')

parser.add_argument('--saved_encoder', type=str, default=r'C:\Users\WB\PycharmProjects\beng280a_pj1\checkpoints\psp\best_model(encoder)_spp@epoch_1.pth')
parser.add_argument('--saved_decoder', type=str, default=r'C:\Users\WB\PycharmProjects\beng280a_pj1\checkpoints\psp\best_model(decoder)_spp@epoch_1.pth')

# device
parser.add_argument('--gpuid', type=int, default=0, help='GPU id to use.')
parser.add_argument('--workers', type=int, default=0, help='number of workers')

opt = parser.parse_args()

ROOT = r"C:\Users\WB\PycharmProjects\beng280a_pj1" # Change to your working folder

TEST_PATH = r'input\covid19-ct-scans\metadata_test.csv'

def test(test_loader, encoder, decoder, device):
    accmeter = []
    ioumeter_bg = []
    ioumeter_lung = []
    ioumeter_infect = []
    encoder.eval().to(device)
    decoder.eval().to(device)

    pred_record = torch.zeros([900, 512, 512])
    label_record = torch.zeros([900, 512, 512])
    img_record = torch.zeros([900, 512, 512])

    with torch.no_grad():
        for i, (img, label_oh) in enumerate(test_loader):
            B = img.size(0)
            img = img.to(device)
            label = torch.argmax(label_oh, 1).to(device)

            # forward
            x1, x2, x3, x4, x5 = encoder(img)
            pred = decoder(img, x1, x2, x3, x4, x5)

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

    #plot_idx = np.argmax(ioumeter_lung)
    #plot_idx = 66

    # test_gt = label_record[plot_idx,:,:].numpy()
    # test_pred = pred_record[plot_idx,:,:].numpy()
    for plot_idx in range(np.size(ioumeter_lung,0)):
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        #fig.suptitle('Accuracy: %.4f; Mean IoU (excluding background): %.4f' % (accmeter[best_idx], miou_nobg_meter[best_idx]))
        ax1.imshow(img_record[plot_idx,:,:],cmap='gray')
        ax1.set_title('CT scan')
        ax2.imshow(label_record[plot_idx,:,:].long())
        ax2.set_title('Ground Truth')
        ax3.imshow(pred_record[plot_idx,:,:])
        ax3.set_title('Prediction')
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

    # loading trained model parameters
    if opt.saved_encoder:
        if os.path.isfile(opt.saved_encoder):
            print("=> loading pretrained encoder: '{}'".format(opt.saved_encoder))
            print("=> loading pretrained decoder: '{}'".format(opt.saved_decoder))
            checkpoint_encoder = torch.load(opt.saved_encoder)
            checkpoint_decoder = torch.load(opt.saved_decoder)
            encoder.load_state_dict(checkpoint_encoder, strict=False)
            decoder.load_state_dict(checkpoint_decoder, strict=False)
            del checkpoint_encoder
            del checkpoint_decoder
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(opt.saved_encoder))


    test_set = COVID19(os.path.join(ROOT, TEST_PATH), data_transform=True)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=True)

    test(test_loader, encoder, decoder, device)


if __name__ == '__main__':
    main()