import pado
import torch
from utils.utils import *
from utils.optics_utils import *
import numpy as np

def simulate_psf(DOE_phase, args, depth=None, propagator = 'ASM', optics_gap = 0, zero_right = True, quantization = 0, simulate_psf_r = False):
    param = args.param
    mask_l, mask_r, mask_half_aperture, aperture_mask = param.mask_l, param.mask_r, param.mask_half_aperture, param.aperture_mask
    if depth is None:
        depth = [param.object_dist]

    DOE_phase = DOE_phase % (2 * np.pi)

    if zero_right:
        DOE_phase = DOE_phase * mask_half_aperture.to(args.device)

    if quantization > 0:
        DOE_phase =  DOE_phase % (2 * np.pi)
        step = 2 * np.pi / quantization
        DOE_phase = floor_DA(DOE_phase / step) * step
    
    doe = pado.DOE(param.R, param.R, param.DOE_pitch, param.material, param.DOE_wvl,args.device, phase =  DOE_phase)
    prop = pado.Propagator(propagator)

    psf_left = []
    psf_right = []
    for d in depth:
        psfs_l = []
        psfs_r = []
        for wvl in param.wvls:
            light = pado.Light(param.R, param.R, param.DOE_pitch, wvl, args.device,B=1)
            if propagator != "Fraunhofer":
                light.set_spherical_light(d)

            doe.change_wvl(wvl)
            light = doe.forward(light)
            if optics_gap > 0:
                print('simulate gap between DOE and thin lens')
                prop_ = pado.Propagator('ASM')
                light = prop_.forward(light, optics_gap)

            if propagator != "Fraunhofer":
                lens = pado.RefractiveLens(param.R, param.R, param.DOE_pitch, param.focal_length, wvl,args.device)
                light = lens.forward(light)

            aperture = pado.Aperture(param.R, param.R, param.DOE_pitch, param.aperture_diamter, param.aperture_shape, wvl, args.device)
            light = aperture.forward(light) 

            psf_size =int(param.R / param.DOE_sample_ratio)
            if psf_size< param.target_PSF_R:
                wl, wr = compute_pad_size(psf_size, param.target_PSF_R)
                padl = param.DOE_sample_ratio*wl
                padr = param.DOE_sample_ratio*wr
                light.pad((padl, padr, padl, padr), padval=0) 
            
            if light.R > param.R:
                mask_l, mask_r = half_aperture_mask(int(light.R/2), light.R-int(light.R/2),light.R)
                aperture_mask = circle_mask(int(light.R/2),int(light.R/2))

            light_l = light.clone()
            light_l.set_amplitude(light_l.get_amplitude() * mask_l.to(args.device))  
            light_l = prop.forward(light_l, param.sensor_dist)
            psf_l = light_l.get_intensity()  

            if simulate_psf_r:
                light_r = light.clone()
                light_r.set_amplitude(light_r.get_amplitude() * mask_r.to(args.device))  
                light_r = prop.forward(light_r, param.sensor_dist)     
                psf_r = light_r.get_intensity()  
            else:
                psf_r = torch.zeros_like(psf_l)
            
            psf = torch.cat([psf_l,psf_r], dim = 0)
            psf = sample_psf(psf, param.DOE_sample_ratio)

            if psf_size > param.target_PSF_R:
                wl, wr = compute_pad_size(param.target_PSF_R, psf_size) 
                psf = psf[:,:,wl:-wr, wl:-wr]  
            
            psf /= (2* torch.sum(psf,(1,2,3), keepdim=True)) 
            psfs_l.append(psf[:1])
            psfs_r.append(psf[1:]) 
        
        psf_left.append(torch.cat(psfs_l, dim = 1))
        psf_right.append(torch.cat(psfs_r, dim = 1))

    height_map = doe.get_height() * aperture_mask.to(args.device)

    return torch.cat(psf_left, dim=0), torch.cat(psf_right, dim=0), height_map

def image_formation(psf_left, psf_right, scene, args, depthmap=None):
    captures_left = []
    captures_right = []
    param = args.param
    padding = int(param.camera_resolution/2)

    # depth application
    if depthmap is not None:
        captures_left_weighted = []
        captures_right_weighted = []
        weight_sums = torch.zeros_like(depthmap)

        # First pass: calculate weights and sum them
        for j in range(len(param.depth)):
            d = param.depth[j]
            depthlayer = depthmap - d
            depth_weights = gaussian_weighting(depthlayer)
            weight_sums += depth_weights

        # Second pass: apply normalized weights and accumulate captures
        for j in range(len(param.depth)):
            d = param.depth[j]
            depthlayer = depthmap - d
            depth_weights = gaussian_weighting(depthlayer) / (weight_sums + 1e-6)
            capture_left = []
            capture_right = []

            for i in range(len(param.wvls)):
                capture_left.append(pado.conv_fft(real2complex(scene[:, i, :, :]), real2complex(psf_left[j, i, :, :]), (padding, padding, padding, padding)).get_mag())
                capture_right.append(pado.conv_fft(real2complex(scene[:, i, :, :]), real2complex(psf_right[j, i, :, :]), (padding, padding, padding, padding)).get_mag())

            capture_left = torch.cat(capture_left, 0)[None, ...]
            capture_right = torch.cat(capture_right, 0)[None, ...]

            # Weight the captures by the normalized depth weights
            captures_left_weighted.append(capture_left * depth_weights)
            captures_right_weighted.append(capture_right * depth_weights)

        # Take a weighted sum of the layers
        capture_enc = torch.sum(torch.cat(captures_left_weighted, 0), 0, keepdim=True)
        capture_ref = torch.sum(torch.cat(captures_right_weighted, 0), 0, keepdim=True)

    # fast approximation for HS application
    elif args.application == 'HS' and args.PSF_file is not None:
        capture_left = torch.zeros_like(scene)
        for i in range(len(param.wvls)):
            copies = args.psf_approx[i]
            for (loc, intensity) in copies:
                shift = loc - int(param.camera_resolution / 2)
                copy = scene[0,i] * intensity
                H, W = copy.shape
                zero_pad = torch.zeros(H, np.abs(shift), device=copy.device)
                if shift >= 0:
                    remaining_part = copy[:, shift:]
                    shifted_copy = torch.cat([remaining_part, zero_pad], dim=-1)
                else:
                    remaining_part = copy[:, :shift]
                    shifted_copy = torch.cat([zero_pad, remaining_part], dim=-1)
                capture_left[0,i] += shifted_copy
        capture_right = scene / 2
        capture_enc = args.param.QE_conv_layer(capture_left)
        capture_ref = args.param.QE_conv_layer(capture_right)

    # fast approximation for HDR application
    elif args.application == 'HDR' and not args.train_optics:
        capture_left = []
        for i in range(len(param.wvls)):
            capture_left.append(pado.conv_fft(real2complex(scene[:,i,:,:]), real2complex(psf_left[:,i,:,:]), (padding,padding,padding,padding)).get_mag())
        capture_left = torch.cat(capture_left,0)[None,...]
        capture_right = scene / 2
        capture_enc = capture_left / (2**args.ND_filter)
        capture_ref = capture_right

    else:
        capture_left = []
        capture_right = []
        for i in range(len(param.wvls)):
            capture_left.append(pado.conv_fft(real2complex(scene[:,i,:,:]), real2complex(psf_left[:,i,:,:]), (padding,padding,padding,padding)).get_mag())
            capture_right.append(pado.conv_fft(real2complex(scene[:,i,:,:]), real2complex(psf_right[:,i,:,:]), (padding,padding,padding,padding)).get_mag())
        capture_left = torch.cat(capture_left,0)[None,...]
        capture_right = torch.cat(capture_right,0)[None,...]
        if args.application == 'HS':
            capture_enc = torch.nn.functional.conv2d(capture_left.type_as(scene), args.param.QE_weight)
            capture_ref = torch.nn.functional.conv2d(capture_right.type_as(scene), args.param.QE_weight)
        elif args.application == 'HDR':
            capture_enc = capture_left / (2**args.ND_filter)
            capture_ref = capture_right
        else:
            capture_enc = capture_left
            capture_ref = capture_right

    if args.sensor_noise > 0:
        capture_enc += (torch.rand(capture_enc.shape) * 2 * args.sensor_noise - args.sensor_noise).type_as(capture_enc)
        capture_ref += (torch.rand(capture_ref.shape) * 2 * args.sensor_noise - args.sensor_noise).type_as(capture_ref)
        capture_enc = torch.clamp(capture_enc,0,None)
        capture_ref = torch.clamp(capture_ref,0,None)

    return torch.clamp(capture_enc + capture_ref,0,1).type_as(scene), torch.clamp(capture_ref,0,1).type_as(scene), capture_enc