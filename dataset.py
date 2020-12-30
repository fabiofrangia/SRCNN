import torch.utils.data as data
import torch 
from torch.utils.data import Dataset
import os
import logging 
from glob import glob
from PIL import Image
import numpy as np

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, mask_suffix='', transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir

        self.mask_suffix = mask_suffix

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.transform = transform

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        
        img = np.array(img)
        mask = np.array(mask)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)


        return {
            'image': (img).type(torch.FloatTensor),
            'mask': (mask).type(torch.FloatTensor)
        }