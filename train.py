import numpy as np
import os
import argparse
from importlib.machinery import SourceFileLoader

import torch
import torchvision
import torch.backends.cudnn as cudnn

from tqdm import trange

from utils.init import init_training
from utils.utils import *
from utils.optics_utils import *
from models.forward import *
from utils.loss import *
from train_step import *

# torch.autograd.set_detect_anomaly(True)

def train_step(batch_data, DOE_phase, optics_optimizer, G, G_optimizer, args):
    param = args.param

    if args.train_optics:
        if args.DOE_phase_noise_scale > 0:
            DOE_phase_noise = (torch.rand((1,1,param.R, param.R)) * args.DOE_phase_noise_scale * 2 - args.DOE_phase_noise_scale).to(args.device)
        else:
            DOE_phase_noise = torch.zeros_like(DOE_phase)
        psf_l, psf_r, height_map = simulate_psf(DOE_phase + DOE_phase_noise, args, depth=param.depth, propagator = args.propagator, optics_gap = 0, \
                                                zero_right = True, quantization = 0, simulate_psf_r = args.train_G)
    if args.train_G:
        scene = batch_data['image'].to(args.device)

        if args.application == 'Depth':
            depthmap = batch_data['depthmap'].to(args.device)
        else:
            depthmap = None

        if args.application == 'HS':
            scene_bgr = args.param.QE_conv_layer(scene)

        if args.train_optics:
            frame1, frame2, enc_preclamp = image_formation(torch.nn.functional.pad(psf_l, (args.edge_padding,args.edge_padding,args.edge_padding,args.edge_padding)), torch.nn.functional.pad(psf_r, (args.edge_padding,args.edge_padding,args.edge_padding,args.edge_padding)), scene, args, depthmap)        
        else:
            with torch.no_grad(): 
                frame1, frame2, enc_preclamp = image_formation(torch.nn.functional.pad(args.psf_l, (args.edge_padding,args.edge_padding,args.edge_padding,args.edge_padding)), torch.nn.functional.pad(args.psf_r, (args.edge_padding,args.edge_padding,args.edge_padding,args.edge_padding)), scene, args, depthmap)
                
        capture_ref = frame2
        capture_enc = torch.clamp(enc_preclamp,0,1).type_as(scene)

    psf_loss = torch.from_numpy(np.array(0)).to(args.device)
    fab_reg = torch.from_numpy(np.array(0)).to(args.device)
    recon_loss = torch.from_numpy(np.array(0)).to(args.device)

    if args.train_optics:
        if 'E2E' not in args.PSF_design and 'HDR' not in args.PSF_design:
            psf_loss = PSF_loss(psf_l, args.psf_target, args.psf_target_is_mask, args.PSF_loss_weight)
            fab_reg = single_pillar_panelty(height_map, args.fab_reg_weight)
            loss = psf_loss + fab_reg
            loss.backward(retain_graph=True)
            optics_optimizer.step()
            optics_optimizer.zero_grad()
        elif args.PSF_design == 'HDR_streak':
            psf_shape_reg = PSF_loss(psf_l, args.psf_target, args.psf_target_is_mask, 0.1)
            max_intensity_loss = torch.max(psf_l) * args.PSF_loss_weight
            fab_reg = single_pillar_panelty(height_map, args.fab_reg_weight)
            loss = psf_shape_reg + max_intensity_loss + fab_reg
            loss.backward(retain_graph=True)
            optics_optimizer.step()
            optics_optimizer.zero_grad()

    if args.train_G:
        if not args.train_optics:
            psf_l = args.psf_l
        if args.application == 'Depth':
            out_depth, depth_TV_loss, depth_loss, recon_loss = train_step_depth(capture_enc, capture_ref, depthmap, psf_l, G, args)        

            recon_loss.backward(retain_graph=True)
            G_optimizer.step()
            G_optimizer.zero_grad()
        elif args.application == 'HDR' :
            out_image, image_loss, highlight_loss, recon_loss = train_step_HDR(capture_enc, capture_ref, scene, psf_l, G, args)

            if 'E2E' in args.PSF_design: 
                psf_loss = args.PSF_loss_weight * torch.sum(((scene < 1) * enc_preclamp)[enc_preclamp >= 1])
                fab_reg = single_pillar_panelty(height_map, args.fab_reg_weight)
                loss = psf_loss + fab_reg + recon_loss.clone()
                if args.psf_target is not None:
                    psf_shape_reg = PSF_loss(psf_l, args.psf_target, args.psf_target_is_mask, 0.1)
                    loss += psf_shape_reg 
                loss.backward(retain_graph=True)
                optics_optimizer.step()
                optics_optimizer.zero_grad()    
            recon_loss.backward(retain_graph=True)
            G_optimizer.step()
            G_optimizer.zero_grad()   

        elif args.application == 'HS':
            out_spectral, out_bgr, recon_loss, spectral_loss, rgb_loss = train_step_HS(capture_enc, capture_ref, scene, scene_bgr, psf_l, G, args)
            recon_loss.backward(retain_graph=True)
            G_optimizer.step()
            G_optimizer.zero_grad()   
        else:
            assert False, "Todo"

    return psf_loss.detach(), fab_reg.detach(), recon_loss.detach()

def log(batch_data, DOE_phase, G, total_step, args):
    param = args.param

    psf_loss = torch.from_numpy(np.array(0)).to(args.device)
    fab_reg = torch.from_numpy(np.array(0)).to(args.device)
    recon_loss = torch.from_numpy(np.array(0)).to(args.device)   

    with torch.no_grad(): 
        if args.train_optics:
            if args.DOE_phase_noise_scale > 0:
                DOE_phase += (torch.rand(DOE_phase.shape) * args.DOE_phase_noise_scale * 2 - args.DOE_phase_noise_scale).to(args.device)
            psf_l, psf_r, height_map = simulate_psf(DOE_phase, args, depth=param.depth, propagator = args.propagator, optics_gap = param.optics_gap, \
                                                    zero_right = True, quantization = args.quantization, simulate_psf_r = True)
            args.psf_l = psf_l
            args.psf_r = psf_r
        if args.train_G:
            scene = batch_data['image'].to(args.device)
            if args.application == 'Depth':
                depthmap = batch_data['depthmap'].to(args.device)
            else:
                depthmap = None
            if args.application == 'HS':
                scene_bgr = args.param.QE_conv_layer(scene)
            frame1, frame2, enc_preclamp = image_formation(torch.nn.functional.pad(args.psf_l, (args.edge_padding,args.edge_padding,args.edge_padding,args.edge_padding)), torch.nn.functional.pad(args.psf_r, (args.edge_padding,args.edge_padding,args.edge_padding,args.edge_padding)), scene, args, depthmap)
            capture_ref = frame2
            capture_enc = torch.clamp(enc_preclamp,0,1).type_as(scene)
            

        if args.train_optics:
            if 'E2E' not in args.PSF_design and 'HDR' not in args.PSF_design:
                psf_loss = PSF_loss(psf_l, args.psf_target, args.psf_target_is_mask, args.PSF_loss_weight)
                fab_reg = single_pillar_panelty(height_map, args.fab_reg_weight)
            elif args.PSF_design == 'HDR_streak':
                psf_shape_reg = PSF_loss(psf_l, args.psf_target, args.psf_target_is_mask, 0.1)
                max_intensity_loss = torch.max(psf_l) * args.PSF_loss_weight
                fab_reg = single_pillar_panelty(height_map, args.fab_reg_weight)

        if args.train_G:
            if not args.train_optics:
                psf_l = args.psf_l
            if args.application == 'Depth':
                out_depth, depth_TV_loss, depth_loss, recon_loss = train_step_depth(capture_enc, capture_ref, depthmap, psf_l, G, args)
                        
            elif args.application == 'HDR' :
                out_image, image_loss, highlight_loss, recon_loss = train_step_HDR(capture_enc, capture_ref, scene, psf_l, G, args)
                if 'E2E' in args.PSF_design:   
                    max_intensity_loss = args.PSF_loss_weight * torch.sum(((scene < 1) * enc_preclamp)[enc_preclamp >= 1])
                    fab_reg = single_pillar_panelty(height_map, args.fab_reg_weight)
                    if args.psf_target is not None:
                        psf_shape_reg = PSF_loss(psf_l, args.psf_target, args.psf_target_is_mask, 0.1)
            elif args.application == 'HS':
                with torch.no_grad(): 
                    out_spectral, out_bgr, recon_loss, spectral_loss, rgb_loss = train_step_HS(capture_enc, capture_ref, scene, scene_bgr, psf_l, G, args)

    if args.train_optics:
        psfs_l, log_psfs_l = plot_psf_list(torch.split(psf_l, 1, 0))
        psfs_r, log_psfs_r = plot_psf_list(torch.split(psf_r, 1, 0))
        psfs = torch.cat([psfs_l,psfs_r],2)
        log_psfs = torch.cat([log_psfs_l,log_psfs_r],2)

        B, C, H, W = psfs.shape
        if C > 3:
            assert B == 1
            psfs = psfs.reshape((C, B, H, W))
            log_psfs = psfs.reshape((C, B, H, W))
            
        if args.psf_target is not None:
            if total_step == 0:
                if C > 3:
                    psf_target, log_psf_target = plot_psf_list(torch.split(args.psf_target.permute(1,0,2,3), 1, 0))
                else:
                    psf_target, log_psf_target =  plot_psf_list(torch.split(args.psf_target, 1, 0))
                
                args.writer.add_image('target_PSF', torchvision.utils.make_grid(psf_target, 8), total_step)

        if 'streak' in args.PSF_design:
            args.writer.add_scalar('val_loss/psf_shape_reg',psf_shape_reg, total_step)
            args.writer.add_scalar('val_loss/psf_max_intensity_loss',max_intensity_loss, total_step)
        else:
            args.writer.add_scalar('val_loss/psf_loss',psf_loss, total_step)
        args.writer.add_scalar('val_loss/fab_reg', fab_reg, total_step)
        args.writer.add_image('PSF', torchvision.utils.make_grid(psfs, 8), total_step)
        args.writer.add_image('LogPSF', torchvision.utils.make_grid(log_psfs,8), total_step)
        args.writer.add_image('Phase', (DOE_phase[0].detach().cpu().numpy() % (2 * np.pi))/ (2 * np.pi), total_step)
    
    if args.train_G:
        if total_step == 0:
            if args.application != 'HS':
                args.writer.add_image('scene', torch.clamp(scene[:,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding],0,1)[0], total_step)
            if (not args.train_optics):
                psfs_l, log_psfs_l = plot_psf_list(torch.split(args.psf_l, 1, 0))
                psfs_r, log_psfs_r = plot_psf_list(torch.split(args.psf_r, 1, 0))
                psfs = torch.cat([psfs_l,psfs_r],2)
                log_psfs = torch.cat([log_psfs_l,log_psfs_r],2)
                B, C, H, W = psfs.shape
                if C > 3:
                    assert B == 1
                    psfs = psfs.reshape((C, B, H, W))
                    log_psfs = psfs.reshape((C, B, H, W))
                args.writer.add_image('PSF', torchvision.utils.make_grid(psfs, 6), total_step)
                args.writer.add_image('LogPSF', torchvision.utils.make_grid(log_psfs,6), total_step)  

        if args.application == 'Depth':
            args.writer.add_scalar('val_loss/depth_TV_loss',depth_TV_loss, total_step)
            args.writer.add_scalar('val_loss/depth_loss',depth_loss, total_step)
            args.writer.add_scalar('val_loss/recon_loss',recon_loss, total_step)
            args.writer.add_image('Recon_depth', out_depth[0]/args.depth_max, total_step)
            args.writer.add_text('depth_range', str([torch.min(out_depth).detach().cpu().numpy(), torch.max(out_depth).detach().cpu().numpy()]), total_step)
            if total_step == 0: 
                args.writer.add_image('capture_enc', 2*capture_enc[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding], total_step)
                args.writer.add_image('capture_ref', 2*capture_ref[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding], total_step)
                args.writer.add_image('depthmap', depthmap[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding]/args.depth_max, total_step)
                args.writer.add_text('depthmap_range', str([torch.min(depthmap).detach().cpu().numpy(), torch.max(depthmap).detach().cpu().numpy()]), total_step)

        elif args.application == 'HDR':
            args.writer.add_scalar('val_loss/image_loss',image_loss, total_step)
            args.writer.add_scalar('val_loss/highlight_loss',highlight_loss, total_step)
            args.writer.add_scalar('val_loss/recon_loss',recon_loss, total_step)
            
            args.writer.add_image('LDR_recon', torch.clamp(out_image[0],0,1), total_step)
            args.writer.add_image('HDR_recon', out_image[0]/64, total_step)
            if total_step == 0: 
                args.writer.add_image('capture_enc', capture_enc[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding], total_step)
                args.writer.add_image('capture_ref', capture_ref[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding], total_step) 
                args.writer.add_image('LDR_scene', torch.clamp(scene[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding],0,1), total_step)
                args.writer.add_image('HDR_scene', torch.clamp(scene[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding]/64,0,1), total_step)
            elif 'E2E' in args.PSF_design:
                args.writer.add_image('capture_enc', capture_enc[0,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding], total_step)

        elif args.application == 'HS':
            B, C, H, W = out_spectral.shape
            args.writer.add_scalar('val_loss/spectral_loss',spectral_loss, total_step)
            args.writer.add_scalar('val_loss/rgb_loss',rgb_loss, total_step)
            out_spectral = out_spectral.reshape((C, B, H, W))
            args.writer.add_image('Recon_image', torchvision.utils.make_grid(out_spectral, 7), total_step)
            args.writer.add_image('Recon_rgb', out_bgr[0,[2,1,0]], total_step)

            if total_step == 0: 
                B, C, H, W = scene.shape
                scene = scene.reshape((C, B, H, W))[:,:,args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding]
                args.writer.add_image('scene', torchvision.utils.make_grid(scene, 7), total_step)
                args.writer.add_image('scene_rgb', scene_bgr[0,[2,1,0],args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding], total_step)
                psf_bgr = args.param.QE_conv_layer(args.psf_l.float())
                psf_bgr /= torch.max(psf_bgr)
                args.writer.add_image('PSF_rgb', psf_bgr[0,[2,1,0]], total_step)

                args.writer.add_image('capture_enc', 2*capture_enc[0,[2,1,0],args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding], total_step)
                args.writer.add_image('capture_ref', 2*capture_ref[0,[2,1,0],args.edge_padding:-args.edge_padding,args.edge_padding:-args.edge_padding], total_step)

def train(args):
    
    # set random seed-----------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled=True
    param = args.param

    trainloader, testloader, test_data, DOE_phase_1D, DOE_phase, optics_optimizer, G, G_optimizer = init_training(args)

    train_psf_loss = 0
    train_fab_reg = 0
    train_recon_loss = 0
    total_step = 0

    for epoch_cnt in trange(args.n_epochs, desc="Epoch"):
        if args.train_G:
            for _, batch_data in enumerate(trainloader):
                if total_step % args.save_freq == 0:
                    torch.save(G.state_dict(), os.path.join(args.result_path,'G_%03d.pt' % (total_step//args.save_freq)))
                    if args.train_optics:
                        if param.rotational_design:
                            torch.save(DOE_phase_1D, os.path.join(args.result_path, 'DOE_phase1D_%03d.pt' % ((total_step+ 1) // args.save_freq)))
                        else:
                            torch.save(DOE_phase, os.path.join(args.result_path,'DOE_phase_%03d.pt' % ((total_step + 1)//args.save_freq)))

                if total_step % args.log_freq == 0:
                    log(test_data, DOE_phase, G, total_step, args)
                    if total_step > 0:
                        args.writer.add_scalar('train_loss/psf_loss',train_psf_loss/args.log_freq, total_step)
                        args.writer.add_scalar('train_loss/fab_reg',train_fab_reg/args.log_freq, total_step)
                        args.writer.add_scalar('train_loss/recon_loss',train_recon_loss/args.log_freq, total_step)
                        train_psf_loss = 0
                        train_fab_reg = 0
                        train_recon_loss = 0

                step_psf_loss, step_fab_reg, step_recon_loss = train_step(batch_data, DOE_phase, optics_optimizer, G, G_optimizer, args)
                if param.rotational_design and args.train_optics:
                    DOE_phase = DOE_1Dto2D(DOE_phase_1D, args)
                train_psf_loss += step_psf_loss
                train_fab_reg += step_fab_reg
                train_recon_loss += step_recon_loss
                total_step += 1
        else:
            for _ in range(args.save_freq):
                if total_step % args.save_freq == 0:
                    if param.rotational_design:
                        torch.save(DOE_phase_1D, os.path.join(args.result_path, 'DOE_phase1D_%03d.pt' % ((total_step+ 1) // args.save_freq)))
                    else:
                        torch.save(DOE_phase, os.path.join(args.result_path,'DOE_phase_%03d.pt' % ((total_step + 1)//args.save_freq)))

                if total_step % args.log_freq == 0:
                    log(test_data, DOE_phase, G, total_step, args)
                    if total_step > 0:
                        args.writer.add_scalar('train_loss/psf_loss',train_psf_loss/args.log_freq, total_step)
                        args.writer.add_scalar('train_loss/fab_reg',train_fab_reg/args.log_freq, total_step)
                        args.writer.add_scalar('train_loss/recon_loss',train_recon_loss/args.log_freq, total_step)
                        train_psf_loss = 0
                        train_fab_reg = 0
                        train_recon_loss = 0

                step_psf_loss, step_fab_reg, step_recon_loss = train_step(None, DOE_phase, optics_optimizer, G, G_optimizer, args)
                if param.rotational_design and args.train_optics:
                    DOE_phase = DOE_1Dto2D(DOE_phase_1D, args)
                train_psf_loss += step_psf_loss
                train_fab_reg += step_fab_reg
                train_recon_loss += step_recon_loss
                total_step += 1

    log(test_data, DOE_phase, G, total_step + 1, args)
    if args.train_G:
        torch.save(G.state_dict(), os.path.join(args.result_path,'G_%03d.pt' % (total_step//args.save_freq)))
    if args.train_optics:
        if param.rotational_design:
            torch.save(DOE_phase_1D, os.path.join(args.result_path, 'DOE_phase1D_%03d.pt' % ((total_step+ 1) // args.save_freq)))
        else:
            torch.save(DOE_phase, os.path.join(args.result_path,'DOE_phase_%03d.pt' % ((total_step + 1)//args.save_freq)))
    

def main():
    parser = argparse.ArgumentParser(
        description='DualPixel Sensor',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    def str2bool(v):
        assert(v == 'True' or v == 'False')
        return v.lower() in ('true')

    def none_or_str(value):
        if value.lower() == 'none':
            return None
        return value

    parser.add_argument('--debug', action="store_true", help='debug mode, train on validation data to speed up the process')
    parser.add_argument('--eval', action="store_true", help='eval mode, skip creating tensorbaord or save scripts')
    parser.add_argument('--train_optics', action="store_true", help='optimize optics design')    
    parser.add_argument('--train_G', action="store_true", help='optimize reconstruction algorithm')
    parser.add_argument('--application', required=True, type=str, choices=['HDR','HS','Depth','Bayer'],help='target application')
    
    parser.add_argument('--pretrained_DOE', default = None, type =none_or_str, help = 'use a pretrained DOE')
    parser.add_argument('--PSF_file', default = None, type =none_or_str, help = 'use experimentally measured PSF')
    parser.add_argument('--pretrained_G', default = None, type =none_or_str, help = 'use a pretrained G')
    parser.add_argument('--result_path', default = './test', type=str, help='dir to save models and checkpoints')
    parser.add_argument('--param_file', default= 'param.py', type=str, help='path to param file')

    parser.add_argument('--n_epochs', default = 10, type = int, help = 'max num of training epoch')
    parser.add_argument('--optics_lr', default=1e-3, type=float, help='optical element learning rate')
    parser.add_argument('--G_lr', default=1e-4, type=float, help='optical element learning rate')

    parser.add_argument('--propagator', default = 'Fresnel', type=str, help = 'define propogator, Fresnel, Fraunhofer or AngularSpectrum')
    parser.add_argument('--quantization', default=16, type=int, help = 'simulate PSF after n-level quantization, 0 means no quantization')
    parser.add_argument('--psf_loss_radius', default = 1, type = int, help = 'radius for loss on PSF')
    parser.add_argument('--fab_reg_weight', default = 0, type = float, help = 'weight for avoiding single pillar')
    parser.add_argument('--DOE_phase_noise_scale', default = 0, type = float, help = 'noise added to DOE phase during training')
    parser.add_argument('--sensor_noise', default=0.01, type=float, help='sensor random noise factor')

    parser.add_argument('--PSF_loss_weight', default = 0, type = float, help = 'weight for loss on PSF')
    parser.add_argument('--PSF_design', default = 'None', type = str, help = 'target PSF design')
    parser.add_argument('--edge_padding', default = 64, type = int, help= 'padding/cropping at the image edge to avoid artifact')
    parser.add_argument('--recon_loss_weight', default = 0, type = float, help = 'weight for loss on reconstruction')

    # Depth 

    # HDR 
    parser.add_argument('--ND_filter', default = 1, type = int, help = 'ND filter power')
    # HS
    
    parser.add_argument('--log_freq', default=100, type=int, help = 'frequency (num_steps) of logging')
    parser.add_argument('--save_freq', default=200, type=int, help = 'frequency (num_steps) of saving checkpoint and visual performance')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    param = SourceFileLoader("param", args.param_file).load_module()
    if 'E2E' in args.PSF_design:
        assert args.train_optics and args.train_G, 'Check training flags'
    
    save_settings(args, param)
    
    train(args)

if __name__ == '__main__':
    
    main()