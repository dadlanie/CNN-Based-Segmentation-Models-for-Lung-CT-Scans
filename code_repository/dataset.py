import cv2
import os
import nibabel as nib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

ROOT = r'C:\Users\WB\PycharmProjects\beng280a_pj1' # Change to your work folder

def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)


class COVID19(Dataset):
    def __init__(self, metadata_path, data_transform=False):
        self.metadata_path = metadata_path
        self.data_transforms = data_transform
        self.raw_data = pd.read_csv(self.metadata_path)
        self.img_list = []
        self.label_list = []
        self.root = ROOT
        self.img_transforms = transforms.Compose(
            [transforms.ToPILImage(),
             # transforms.RandomCrop(size=28, padding=4),
             # transforms.RandomHorizontalFlip(),
             transforms.Resize((512, 512)),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        )
        # self.label_transforms = transforms.Compose(
        #     [transforms.ToPILImage(),
        #     transforms.Resize((512, 512))
        #     ]
        # )

        for i in range(self.raw_data.shape[0]):
            img = read_nii(os.path.join(self.root,self.raw_data.loc[i, 'ct_scan']))
            label = read_nii(os.path.join(self.root,self.raw_data.loc[i, 'lung_and_infection_mask']))
            for j in range(int((img.shape[2]) / 3)):
                n = int(j + (img.shape[2]) / 3)
                self.img_list.append(cv2.normalize(src=img[..., n], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
                self.label_list.append(label[..., n])



    def __getitem__(self, idx):
        img = self.img_list[idx]
        # img = cv2.equalizeHist(img)
        label = self.label_list[idx]
        label = cv2.resize(label, (512, 512))


        if self.data_transforms:
            img = self.img_transforms(img)
            # label = self.label_transforms(label)
            label0 = (label == 0)
            label1 = (label == 1)
            label2 = (label == 2)
            label3 = (label == 3)
            label_oh = np.stack((label0, (label1 + label2), label3))
            label_oh = label_oh.astype(np.float32)

        return img, label_oh

    def __len__(self):
        return len(self.img_list)