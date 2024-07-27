import numpy as np
import torch
import shutil
import json
from glob import glob
import os

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

def save_settings(args, param):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    args_dict = vars(args)
    with open(os.path.join(args.result_path,'args.json'), "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)
    shutil.copy(args.param_file, args.result_path)
    if args.pretrained_DOE is not None:
        shutil.copy(last_save(args.pretrained_DOE, 'DOE_phase*'), os.path.join(args.result_path, 'init'))
    if args.pretrained_G is not None:
        shutil.copy(last_save(args.pretrained_G, 'G*'), os.path.join(args.result_path, 'init'))
    args.param = param

def last_save(ckpt_path, file_format):
    return sorted(glob(os.path.join(ckpt_path, file_format)))[-1]

def plot_psf_list(psfs_list, normalize = True):
    psfs = []
    log_psfs = []
    for psf in psfs_list:
        log_psf = torch.log(psf + 1e-9)
        log_psf -= torch.min(log_psf)
        log_psf /= torch.max(log_psf)
        log_psfs.append(log_psf)
        if normalize:
            psfs.append(psf/torch.max(psf))
        else:
            psfs.append(psf)
    return torch.cat(psfs,0), torch.cat(log_psfs,0)

def round_DA(input):
    # This is equivalent to replacing round function (non-differentiable) with
    # an identity function (differentiable) only when backward.
    # https://github.com/kitayama1234/Pytorch-BPDA
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out

def floor_DA(input):
    # This is equivalent to replacing floor function (non-differentiable) with
    # an identity function (differentiable) only when backward.
    # https://github.com/kitayama1234/Pytorch-BPDA
    forward_value = torch.floor(input)
    out = input.clone()
    out.data = forward_value.data
    return out

def gaussian_weighting(depth_diff, sigma=0.075, clip_limit=20):
    scaled_diff = depth_diff / sigma
    scaled_diff = torch.clamp(scaled_diff, -clip_limit, clip_limit)
    return torch.exp(-0.5 * scaled_diff ** 2)

class ChannelRolling(object):
    """Randomly shuffle the color channel
    """

    def __call__(self, image):
        C,H,W = image.shape
        return np.roll(image, np.random.randint(C), axis=0)
    
def splitRaw(raw):
    return raw[::2,::2], raw[0::2,1::2], raw[1::2,0::2], raw[1::2,1::2]

def left_right_calibration(capture_sum, capture_r):
    channels_sum = splitRaw(capture_sum)
    channels_r = splitRaw(capture_r)
    H_dim,W_dim = channels_sum[0].shape
    ratio_fit = []
    for (c_s, c_r) in zip(channels_sum, channels_r):
        data = np.sum(c_r,0)/np.sum(c_s,0)
        spl = UnivariateSpline(np.arange(len(data)), data)
        smoothed_data_spline = spl(np.arange(len(data)))
        ratio_fit.append(smoothed_data_spline)
    
    ratio_expand = np.zeros_like(capture_sum, dtype= 'float64')
    ratio_expand[::2,::2] = np.repeat(ratio_fit[0][None,...],H_dim,0)
    ratio_expand[0::2,1::2] = np.repeat(ratio_fit[1][None,...],H_dim,0)
    ratio_expand[1::2,0::2] = np.repeat(ratio_fit[2][None,...],H_dim,0)
    ratio_expand[1::2,1::2] = np.repeat(ratio_fit[3][None,...],H_dim,0)
        
    return ratio_expand