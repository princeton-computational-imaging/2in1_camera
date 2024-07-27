import torch
import numpy as np
import torchvision.transforms as T
from utils.optics_utils import *

def gradient(img):
    bs_img, c_img, h_img, w_img = img.size()
    g_h = img[:,:,1:,:]-img[:,:,:-1,:]
    g_w = img[:,:,:,1:]-img[:,:,:,:-1]
    return g_h, g_w

def TV_loss(img, weight, args, GT=None):
    if weight == 0:
        return torch.from_numpy(np.array(0)).to(args.device)
    else:
        g_h, g_w = gradient(img)
        if GT is None:
            loss = torch.mean(torch.abs(g_h)) + torch.mean(torch.abs(g_w))
            return loss * weight
        else:
            g_h_gt, g_w_gt = gradient(GT)
            loss = torch.mean(torch.abs(g_h - g_h_gt)) + torch.mean(torch.abs(g_w - g_w_gt))
            return loss * weight

def single_pillar_panelty(height_map, weight, max_height = 1.2e-6):
    threshold = max_height / 16 * 2
    if weight == 0:
        return torch.from_numpy(np.array(0)).to(height_map.device)
    else:
        tv_min = torch.minimum(torch.abs(height_map[:,:,1:,1:]-height_map[:,:,:-1,1:]), torch.abs(height_map[:,:,1:,1:]-height_map[:,:,1:,:-1]))
        tv_cut = (tv_min > threshold) * tv_min
        return weight * torch.mean(tv_cut) / max_height

def target_psf(args):
    param = args.param
    center_x = int(param.target_PSF_R/2)
    if args.PSF_design == 'Depth_ring':
        [x, y] = np.mgrid[-int(center_x):int(param.target_PSF_R - center_x),-int(center_x):int(param.target_PSF_R - center_x)]
        dist = np.sqrt(x**2 +y**2)
        mask = []
        for r in args.param.radius:
            mask.append((torch.tensor(1.0*(dist <= r + args.psf_loss_radius)*(dist >= r - args.psf_loss_radius))[None, None, ...]).to(args.device))
        return torch.cat(mask, dim=0), True
    elif args.PSF_design == 'HS_rainbow':
        C = len(param.wvls)
        mask = torch.zeros((1,C,param.target_PSF_R,param.target_PSF_R))
        for i in range(C):
            idx = center_x - 1 - int(max_diffraction_pxl(param.wvls[i], param.slit_width_pxl*param.DOE_pitch, param.focal_length, param.camera_pitch))
            mask[:,i,center_x,idx] = 1
            mask[:,i,center_x,idx+1] = 1
            mask[:,i,center_x,idx-1] = 1
        return mask.to(torch.float32).to(args.device), True
    elif args.PSF_design == 'Bayer_RGGB':
        mask = torch.zeros((1,3,param.target_PSF_R,param.target_PSF_R))
        mask[:,0,center_x, center_x] = 1
        mask[:,1,center_x+1, center_x] = 1
        mask[:,1,center_x, center_x+1] = 1
        mask[:,2,center_x+1, center_x+1] = 1
        return mask.to(args.device), True
    elif args.PSF_design == 'HDR_streak' or args.PSF_design == 'HDR_E2E_streak':
        mask = torch.zeros((1, 1, param.target_PSF_R,param.target_PSF_R)).to(args.device)
        mask[...,center_x, center_x] = 1
        for i in range(1, center_x):
            for j in range(-1,1):
                mask[...,j + center_x - i,j + center_x] = 1
                mask[...,j + center_x + i,j + center_x] = 1
                mask[...,j + center_x,j + center_x - i] = 1
                mask[...,j + center_x ,j + center_x + i] = 1
        return mask, True
    elif 'E2E' in args.PSF_design:
        return None, None
    else:
        assert False, 'PSF_design undefined:%s' % args.PSF_design

def PSF_loss(psf, target, target_is_mask, weight):
    if weight == 0:
        return torch.from_numpy(np.array(0)).to(psf.device)
    else:
        if target_is_mask:
            target_energy = torch.sum(psf,(2,3))
            psf_energy = torch.sum(psf*target,(2,3))
            return weight * torch.mean(torch.abs(target_energy - psf_energy))
        else:
            R = psf.shape[-1]
            half_one = torch.ones([1,1, R,int(R/2)])
            half_zero = torch.ones([1,1,R,int(R/2)]) * 1e-20
            mask = torch.cat([half_one, half_zero], dim = -1).to(psf.device)
            return weight * torch.max(mask * torch.abs(psf - target))