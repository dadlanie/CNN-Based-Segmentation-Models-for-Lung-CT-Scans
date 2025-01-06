import os
import sys
import random
import math

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from dataset import COVID19

ROOT = r'C:\Users\WB\PycharmProjects\beng280a_pj1' # Change to your working folder

TRAIN_PATH = r'input\covid19-ct-scans\metadata_train.csv'
VAL_PATH = r'input\covid19-ct-scans\metadata_val.csv'

def main():
    train_set = COVID19(os.path.join(ROOT, TRAIN_PATH), data_transform=True)
    val_set = COVID19(os.path.join(ROOT, VAL_PATH), data_transform=True)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    for i, (img, label_oh) in enumerate(train_loader):
        img = img[0,0,:,:]
        label = torch.argmax(label_oh, 1)
        label = label[0,:,:]

        fig = plt.figure(figsize=(9, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='bone')
        plt.title('CT Scan')

        plt.subplot(1, 2, 2)
        plt.imshow(img, cmap='bone')
        plt.imshow(label, alpha=0.5, cmap='nipy_spectral')
        plt.title('Lung and Infection Mask')
        plt.show()
        #fig.savefig(r'C:\Users\WB\Desktop\CT_LUNG_data_visual\img{0}.png'.format(i))
        plt.close(fig)



if __name__ == '__main__':
    main()