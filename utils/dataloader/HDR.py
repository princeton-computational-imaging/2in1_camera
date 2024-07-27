import os
import numpy as np
import torch
from torch.utils.data import Dataset

class HDRi(Dataset):
    """
    Custom HDR dataset that returns a dictionary of LDR input image, HDR ground truth image and file path.
    """

    def __init__(self, mode, clip_min = 0.5, clip_max = 2.5, root='/n/fs/pci-sharedt/HDRData/hdri_npy', transform=None):

        self.mode = mode
        self.dataset_path = root
        
        self.train_images_list = os.path.join(self.dataset_path, "train.txt")
        self.val_images_list = os.path.join(self.dataset_path, "val.txt")

        train = np.loadtxt(self.train_images_list, delimiter=',')
        val = np.loadtxt(self.val_images_list, delimiter=',')

        # paths to LDR and HDR images ->

        self.train_image_names = ['%06d.npy' % int(i) for i in train]
        self.val_image_names = ['%06d.npy' % int(i) for i in val]

        self.images_list = []
        if mode == 'train':
            self.images_list = self.train_image_names
        else:
            self.images_list = self.val_image_names
        
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.transform = transform

    def __getitem__(self, index):

        image_path = os.path.join(self.dataset_path, self.images_list[index])
        image = np.load(image_path)

        # Vary clip limit for data augmentation
        if self.mode == 'train':
            clip_limit = np.random.uniform(self.clip_min,self.clip_max)
        else:
            clip_limit = (self.clip_min + self.clip_max)/2
        sc = np.percentile(image, 100-clip_limit, interpolation='higher')
        image = np.clip(image / sc, 0, 256)
        
        data = torch.from_numpy(image).permute(2,0,1)
        
        if self.transform:
            data = self.transform(data)
            

        sample_dict = {
            "image": data,
            "path": image_path,
        }

        return sample_dict

    def __len__(self):
        return len(self.images_list)