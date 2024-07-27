import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import conv2d
import h5py
import scipy.io as sio
import matplotlib.image as mpimg
from glob import glob

def file_match(filetype, root):

    files = []
    pattern   = "*.%s" %filetype
    for dir,_,_ in os.walk(root):
        files.extend(glob(os.path.join(dir,pattern))) 

    return files

class ICVL(Dataset):
    def __init__(self, root = '/n/fs/pci-sharedt/Array_DOE/data/ICVL', filetype = 'mat', transform=None):
        print("Start loading data from %s" % root)
        # build image list
        self.imglist = file_match(filetype, root)
        self.transform = transform

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        data_temp = h5py.File(self.imglist[index], 'r') 
        image = np.flip(np.array(data_temp['rad']),0).astype(np.float32)
        data = image / np.max(image) 
        data = torch.Tensor(data)
        
        if self.transform:
            data = self.transform(data)

        return {'image': data} 

class HSDB(Dataset):
    def __init__(self, root = '/n/fs/pci-sharedt/Array_DOE/data/hsdb/', filetype = 'mat', transform=None):
        print("Start loading data from %s" % root)
        # build image list
        self.imglist = file_match(filetype, root)
        self.transform = transform

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        #image = np.flip(np.transpose(np.array(data_temp['rad']),[1,2,0]),0).astype(np.float32)
        data_temp = sio.loadmat(self.imglist[index]) 
        image = data_temp['ref'].astype(np.float32)
        image = np.transpose(image, (2,0,1))
        data = image / np.max(image) 
        data = torch.Tensor(data)

        if self.transform:
            data = self.transform(data)

        return {'image': data} 

class CAVE(Dataset):
    def __init__(self, root = '/n/fs/pci-sharedt/Array_DOE/data/CAVE/', filetype = 'mat', transform=None):
        print("Start loading data from %s" % root)
        # build image list
        file_lists = os.listdir(root)
        self.imglist = []
        for file in file_lists:
            image = []
            for i in range(31):
                path = os.path.join(root, file, file, "%s_%02d.png" % (file,i+1))
                assert os.path.exists(path), 'path: %s does not exist'%path
                image.append(path)
            self.imglist.append(image)
        self.transform = transform
        
    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        paths = self.imglist[index]
        image = []
        for path in paths:
            data = mpimg.imread(path)
            if data.ndim == 2:
                image.append(data[...,None])
            elif data.ndim == 3:
                image.append(np.mean(data, axis = -1, keepdims=True))
            else:
                assert False
        image = np.concatenate(image, axis = -1)
        image = np.transpose(image, (2,0,1))
        data = image / np.max(image) 
        data = torch.Tensor(data)

        if self.transform:
            data = self.transform(data)
        return {'image': data, 'path': path} 
