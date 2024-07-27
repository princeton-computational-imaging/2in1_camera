import numpy as np
import os
import argparse
from importlib.machinery import SourceFileLoader

import torch
import torch.backends.cudnn as cudnn

import json

from utils.init import init_training
from utils.utils import *
from utils.optics_utils import *
from models.forward import *
from train_step import *

def eval_step(batch_data, G, args):
    scene = batch_data['image'].to(args.device)
    if args.application == 'Depth':
        depthmap = batch_data['depthmap'].to(args.device)
    else:
        depthmap = None
    if args.application == 'HS':
        with torch.no_grad(): 
            scene_bgr = args.param.QE_conv_layer(scene)
    param = args.param
    
    with torch.no_grad(): 
        frame1, frame2, enc_preclamp = image_formation(torch.nn.functional.pad(args.psf_l,    (args.edge_padding,args.edge_padding,args.edge_padding,args.edge_padding)), torch.nn.functional.pad(args.psf_r, (args.edge_padding,args.edge_padding,args.edge_padding,args.edge_padding)), scene, args, depthmap)  
        capture_ref = frame2
        capture_enc = torch.clamp(enc_preclamp,0,1).type_as(scene)
        

        if args.application == 'HS':
            file_name = batch_data['path'][0].split('/')[-2]
            out_spectral, out_bgr, recon_loss, spectral_loss, rgb_loss = train_step_HS(capture_enc, capture_ref, scene, scene_bgr, args.psf_l, G, args)
            np.save(os.path.join(args.result_path,'HS_GT',file_name), scene[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding].permute(1,2,0).cpu().numpy())
            np.save(os.path.join(args.result_path,'RGB_GT',file_name), scene_bgr[0,[2,1,0],args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding].permute(1,2,0).cpu().numpy())
            np.save(os.path.join(args.result_path,'capture_enc',file_name), capture_enc[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding].permute(1,2,0).cpu().numpy())
            np.save(os.path.join(args.result_path,'capture_ref',file_name), capture_ref[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding].permute(1,2,0).cpu().numpy())
            np.save(os.path.join(args.result_path,'out_spectral',file_name), out_spectral[0].permute(1,2,0).cpu().numpy())
            np.save(os.path.join(args.result_path,'out_rgb',file_name), out_bgr[0,[2,1,0]].permute(1,2,0).cpu().numpy())
        elif args.application == 'HDR' :
            file_name = batch_data['path'][0].split('/')[-1]
            out_image, image_loss, highlight_loss, recon_loss = train_step_HDR(capture_enc, capture_ref, scene, args.psf_l, G, args)  
            np.save(os.path.join(args.result_path,'HDR_GT',file_name), scene[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding].permute(1,2,0).cpu().numpy())
            np.save(os.path.join(args.result_path,'capture_enc',file_name), capture_enc[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding].permute(1,2,0).cpu().numpy())
            np.save(os.path.join(args.result_path,'capture_ref',file_name), capture_ref[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding].permute(1,2,0).cpu().numpy())
            np.save(os.path.join(args.result_path,'output',file_name), out_image[0].permute(1,2,0).cpu().numpy())  
        else:
            file_name = batch_data['id'][0]
            out_depth, depth_TV_loss, depth_loss, recon_loss = train_step_depth(capture_enc, capture_ref, depthmap, args.psf_l, G, args) 
            np.save(os.path.join(args.result_path,'Depth_GT',file_name), depthmap[0,0,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding].cpu().numpy())
            np.save(os.path.join(args.result_path,'capture_enc',file_name), capture_enc[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding].permute(1,2,0).cpu().numpy())
            np.save(os.path.join(args.result_path,'capture_ref',file_name), capture_ref[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding].permute(1,2,0).cpu().numpy())
            np.save(os.path.join(args.result_path,'output',file_name), out_depth[0,0].cpu().numpy())  


def eval(args):
    
    # set random seed-----------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled=True

    param = args.param

    trainloader, testloader, test_data, DOE_phase_1D, DOE_phase, optics_optimizer, G, G_optimizer = init_training(args)
    
    if args.application == 'HDR' :
        if not os.path.exists(os.path.join(args.result_path,'HDR_GT')):
            os.makedirs(os.path.join(args.result_path,'HDR_GT')) 
        if not os.path.exists(os.path.join(args.result_path,'capture_enc')):
            os.makedirs(os.path.join(args.result_path,'capture_enc'))
        if not os.path.exists(os.path.join(args.result_path,'capture_ref')):
            os.makedirs(os.path.join(args.result_path,'capture_ref'))
        if not os.path.exists(os.path.join(args.result_path,'output')):
            os.makedirs(os.path.join(args.result_path,'output'))    
    elif args.application == 'HS' :
        if not os.path.exists(os.path.join(args.result_path,'RGB_GT')):
            os.makedirs(os.path.join(args.result_path,'RGB_GT'))
        if not os.path.exists(os.path.join(args.result_path,'HS_GT')):
            os.makedirs(os.path.join(args.result_path,'HS_GT')) 
        if not os.path.exists(os.path.join(args.result_path,'capture_enc')):
            os.makedirs(os.path.join(args.result_path,'capture_enc'))
        if not os.path.exists(os.path.join(args.result_path,'capture_ref')):
            os.makedirs(os.path.join(args.result_path,'capture_ref'))
        if not os.path.exists(os.path.join(args.result_path,'out_spectral')):
            os.makedirs(os.path.join(args.result_path,'out_spectral'))    
        if not os.path.exists(os.path.join(args.result_path,'out_rgb')):
            os.makedirs(os.path.join(args.result_path,'out_rgb'))   
    elif args.application == 'Depth' :
        if not os.path.exists(os.path.join(args.result_path,'Depth_GT')):
            os.makedirs(os.path.join(args.result_path,'Depth_GT')) 
        if not os.path.exists(os.path.join(args.result_path,'capture_enc')):
            os.makedirs(os.path.join(args.result_path,'capture_enc'))
        if not os.path.exists(os.path.join(args.result_path,'capture_ref')):
            os.makedirs(os.path.join(args.result_path,'capture_ref'))
        if not os.path.exists(os.path.join(args.result_path,'output')):
            os.makedirs(os.path.join(args.result_path,'output'))    
     
    for _, batch_data in enumerate(testloader):
        eval_step(batch_data, G, args)

def main():
    parser = argparse.ArgumentParser(
        description='DualPixel Sensor',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--ckpt_path', default = './test', type=str, help='dir to load models and checkpoints')
   
    args_ = parser.parse_args()
    args = json.load(open(os.path.join(args_.ckpt_path,'args.json'),'r'))
    args = AttributeDict(args)
    param = SourceFileLoader("param", os.path.join(args_.ckpt_path, args.param_file.split('/')[-1])).load_module()
    args.param = param
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.train_optics:
        args.pretrained_DOE = args_.ckpt_path
    args.pretrained_G = args_.ckpt_path
    args.train_optics = False
    args.eval = True
    
    eval(args)

if __name__ == '__main__':
    
    main()