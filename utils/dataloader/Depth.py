import os
import torch
import numpy as np
import imageio
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class SceneFlow(Dataset):

    def __init__(self, root = '/n/fs/pci-sharedt/DualPixel/data/FlyThings3D/', dataset='train', \
                 depth_min=1, depth_max=5, transform=None):

        super().__init__()
        if dataset == 'train':
            image_dir = os.path.join(root,'FlyingThings3D_subset/train/image_clean/right')
            disparity_dir = os.path.join(root,'FlyingThings3D_subset/train/disparity/right')
            self.var = 1
        elif dataset == 'val':
            image_dir = os.path.join(root,'FlyingThings3D_subset/val/image_clean/right')
            disparity_dir = os.path.join(root,'FlyingThings3D_subset/val/disparity/right')
            self.var = 0
        else:
            raise ValueError(f'dataset ({dataset}) has to be "train," "val,"')

        self.sample_ids = []
        for filename in sorted(os.listdir(image_dir)):
            if '.png' in filename:
                id = os.path.splitext(filename)[0]
                disparity_path = os.path.join(disparity_dir, f'{id}.pfm')
                if os.path.exists(disparity_path):
                    sample_id = {
                        'image_dir': image_dir,
                        'disparity_dir': disparity_dir,
                        'id': id,
                    }
                    self.sample_ids.append(sample_id)
                else:
                    print(f'Disparity image does not exist!: {disparity_path}')

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.dataset = dataset
        self.transform = transform

    def stretch_depth(self, depth, depth_min, depth_max):
        depth_range = depth_max - depth_min
        return depth_range * depth + depth_min

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        image_dir = sample_id['image_dir']
        disparity_dir = sample_id['disparity_dir']
        id = sample_id['id']     
        
        disparity = np.flip(imageio.imread(os.path.join(disparity_dir, f'{id}.pfm')), axis=0).astype(np.float32)
        
        # A far object is 0.
        depthmap = disparity
        depthmap -= depthmap.min()
        depthmap /= depthmap.max()

        # Flip the value. A near object is 0.
        depthmap = 1. - depthmap
        
        img = cv2.imread(os.path.join(image_dir, f'{id}.png'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = torch.from_numpy(img).permute(2, 0, 1) # 0-255
        
        depthmap = torch.from_numpy(depthmap)[None, ...] 
        # SceneFlow's depthmap has some aliasing artifact.
        depthmap = transforms.GaussianBlur(kernel_size=(5, 5),sigma=(0.8, 0.8))(depthmap)
        depthmap = self.stretch_depth(depthmap, self.depth_min, self.depth_max)

        data = torch.cat([img,depthmap])
        if self.transform:
            data = self.transform(data)

        img = torch.clamp(data[:3]/255,0,1).float()
        depthmap = data[-1:]
        
        sample = {'id': id, 'image': img, 'depthmap': depthmap}
        return sample