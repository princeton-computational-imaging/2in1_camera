import numpy as np
import torch

from utils.utils import *
from utils.optics_utils import *
from models.forward import *
from utils.loss import *

def train_step_HS(capture_enc, capture_ref, scene, scene_bgr, psf_l, G, args):
    out_spectral = G(capture_enc, capture_ref, torch.nn.functional.pad(psf_l, (args.edge_padding,args.edge_padding,args.edge_padding,args.edge_padding)))
    out_spectral = out_spectral[:,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding]

    out_bgr = args.param.QE_conv_layer(out_spectral)
    spectral_loss = args.recon_loss_weight * args.sam_criterion(out_spectral, scene[:,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding])
    rgb_loss = args.recon_loss_weight * args.perceptual_criterion(2*out_bgr - 1, 2*scene_bgr[:,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding]-1)
    recon_loss = spectral_loss + rgb_loss
    return out_spectral, out_bgr, recon_loss, spectral_loss, rgb_loss

def train_step_depth(capture_enc, capture_ref, depthmap, psf_l, G, args):
    out_depth = G(capture_enc, capture_ref, torch.nn.functional.pad(psf_l, (args.edge_padding,args.edge_padding,args.edge_padding,args.edge_padding)))[:,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding]
    depthmap = depthmap[:,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding]

    depth_TV_loss = TV_loss(out_depth, 0.1*args.recon_loss_weight, args, GT=depthmap)
    depth_loss = args.recon_loss_weight * args.l1_criterion(out_depth, depthmap)
    recon_loss = depth_TV_loss + depth_loss
    return out_depth, depth_TV_loss, depth_loss, recon_loss

def train_step_HDR(capture_enc, capture_ref, scene, psf_l, G, args):
    out_image = G(capture_enc, capture_ref, torch.nn.functional.pad(psf_l, (args.edge_padding,args.edge_padding,args.edge_padding,args.edge_padding)))[:,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding]
    scene = scene[:,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding]
    image_loss = args.recon_loss_weight * args.l1_criterion(out_image, scene)
    mask = scene>2
    if torch.sum(mask) > 0:
        highlight_loss = 0.1 * args.recon_loss_weight * args.l1_criterion(out_image[mask], scene[mask])
    else:
        highlight_loss = torch.from_numpy(np.array(0)).to(args.device)
    recon_loss = image_loss + highlight_loss 

    return out_image, image_loss, highlight_loss, recon_loss