import torch
import numpy as np
import os
import shutil

from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import lpips
from torchmetrics.image import SpectralAngleMapper

from utils.utils import *
from utils.optics_utils import *
from models.forward import *
from utils.loss import *

def init_inference(args):
    # PSF
    if args.PSF_file is None:
        args.DOE_phase_init_ckpt = last_save(args.pretrained_DOE, 'DOE_phase*')
        param.DOE_phase_init = torch.load(args.DOE_phase_init_ckpt, map_location='cpu').detach()
        if param.rotational_design:
            DOE_phase_1D = Variable(param.DOE_phase_init.to(args.device), requires_grad=True)
            optics_optimizer = optim.Adam([DOE_phase_1D], lr=args.optics_lr)
            DOE_phase = DOE_1Dto2D(DOE_phase_1D, args)
        else:
            if param.DOE_phase_init.shape[-1] < param.R:
                param.DOE_phase_init = torch.nn.functional.upsample(param.DOE_phase_init,(param.R,param.R))
            DOE_phase_1D = None
            DOE_phase = Variable(param.DOE_phase_init.to(args.device), requires_grad=True)
            optics_optimizer =optim.Adam([DOE_phase], lr=args.optics_lr)

        with torch.no_grad():
            args.psf_l, args.psf_r, _ = simulate_psf(DOE_phase, args, depth=param.depth, propagator = args.propagator, optics_gap = param.optics_gap, \
                                                         zero_right = True, quantization = args.quantization, simulate_psf_r = True)
    else:
        DOE_phase_1D = DOE_phase = optics_optimizer = None
        assert not args.train_optics
        if args.application != 'Depth':
            psf = torch.from_numpy(np.load(args.PSF_file)).to(args.device).permute(0,3,1,2)
            psf /= (2* torch.sum(psf,(2,3), keepdim=True)) 
            # in simulation, left side is the encoded side, but in the experimental setup, the right side is the encoded side
            args.psf_r = psf[:1]
            args.psf_l = psf[1:]
        else:
            psf = torch.from_numpy(np.load(args.PSF_file)).to(args.device).permute(0,1,4,2,3)
            psf /= (2* torch.sum(psf,(3,4), keepdim=True)) 
            # in simulation, left side is the encoded side, but in the experimental setup, the right side is the encoded side
            args.psf_r = psf[0]
            args.psf_l = psf[1]
            
    if args.application == 'Depth':
        args.depth_min=np.min(args.param.depth)
        args.depth_max=np.max(args.param.depth)
        
        from models.Depth_resnet import Depth_G
        G = Depth_G().to(args.device)    
        G.load_state_dict(torch.load(last_save(args.pretrained_G, 'G*'), map_location='cpu'))
    elif args.application == 'HS':
        args.param.QE_weight = args.param.QE_weight.to(args.device)   
        args.param.QE_conv_layer = torch.nn.Conv2d(in_channels=31, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        args.param.QE_conv_layer.weight = torch.nn.Parameter(args.param.QE_weight)

        if args.PSF_file is not None:
            import pickle
            with open(args.PSF_file.replace('.npy','_approx.pkl'), 'rb') as file:
                args.psf_approx = pickle.load(file)

        from models.HS_DWDN import HS_G
        G = HS_G().to(args.device)
        G.load_state_dict(torch.load(last_save(args.pretrained_G, 'G*'), map_location='cpu'))
        G.to(args.device)
    elif args.application == 'HDR':
        from models.HDR_DWDN import HDR_G
        G = HDR_G().to(args.device)
        G.load_state_dict(torch.load(last_save(args.pretrained_G, 'G*'), map_location='cpu'))
        G.to(args.device)
    return G


def init_training(args):
    param = args.param
    if not args.eval:
        args.writer = SummaryWriter(args.result_path)
    args.l1_criterion = torch.nn.L1Loss().to(args.device)
    args.l2_criterion = torch.nn.MSELoss().to(args.device)
    args.perceptual_criterion = lpips.LPIPS(net='vgg').to(args.device)
    args.sam_criterion = SpectralAngleMapper().to(args.device)

    # PSF
    if args.PSF_file is None:
        if args.pretrained_DOE is not None:
            args.DOE_phase_init_ckpt = last_save(args.pretrained_DOE, 'DOE_phase*')
            param.DOE_phase_init = torch.load(args.DOE_phase_init_ckpt, map_location='cpu').detach()
        if param.rotational_design:
            DOE_phase_1D = Variable(param.DOE_phase_init.to(args.device), requires_grad=True)
            optics_optimizer = optim.Adam([DOE_phase_1D], lr=args.optics_lr)
            DOE_phase = DOE_1Dto2D(DOE_phase_1D, args)
        else:
            if param.DOE_phase_init.shape[-1] < param.R:
                param.DOE_phase_init = torch.nn.functional.upsample(param.DOE_phase_init,(param.R,param.R))
            DOE_phase_1D = None
            DOE_phase = Variable(param.DOE_phase_init.to(args.device), requires_grad=True)
            optics_optimizer =optim.Adam([DOE_phase], lr=args.optics_lr)

        if not args.train_optics:
            with torch.no_grad():
                args.psf_l, args.psf_r, _ = simulate_psf(DOE_phase, args, depth=param.depth, propagator = args.propagator, optics_gap = param.optics_gap, \
                                                         zero_right = True, quantization = args.quantization, simulate_psf_r = True)
        else:
            if args.application == 'Depth' and not args.train_G:
                param.depth = param.depth[[0,-1]]
                param.radius = param.radius[[0,-1]]
            args.psf_target, args.psf_target_is_mask = target_psf(args)
            
    else:
        DOE_phase_1D = DOE_phase = optics_optimizer = None
        assert not args.train_optics
        if args.application != 'Depth':
            psf = torch.from_numpy(np.load(args.PSF_file)).to(args.device).permute(0,3,1,2)
            psf /= (2* torch.sum(psf,(2,3), keepdim=True)) 
            # in simulation, left side is the encoded side, but in the experimental setup, the right side is the encoded side
            args.psf_r = psf[:1]
            args.psf_l = psf[1:]
        else:
            psf = torch.from_numpy(np.load(args.PSF_file)).to(args.device).permute(0,1,4,2,3)
            psf /= (2* torch.sum(psf,(3,4), keepdim=True)) 
            # in simulation, left side is the encoded side, but in the experimental setup, the right side is the encoded side
            args.psf_r = psf[0]
            args.psf_l = psf[1]
    
    # Depth
    if args.application == 'Depth':
        args.depth_min=np.min(args.param.depth)
        args.depth_max=np.max(args.param.depth)

        if args.train_G:
            from utils.dataloader.Depth import SceneFlow
            transform_test = transforms.Compose([
                    transforms.CenterCrop([args.param.camera_resolution+2*args.edge_padding, args.param.camera_resolution+2*args.edge_padding])
                ])
            testset = SceneFlow(dataset = 'val', transform = transform_test, depth_min=args.depth_min, depth_max=args.depth_max)
            testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
            for _, batch_data in enumerate(testloader):
                test_data = batch_data
                break
            if args.debug:
                trainloader = testloader
            else:
                transform_train = transforms.Compose([
                        transforms.RandomCrop([args.param.camera_resolution+2*args.edge_padding, args.param.camera_resolution+2*args.edge_padding],pad_if_needed=True),
                        transforms.RandomHorizontalFlip(),
                    ])
                trainset = SceneFlow(dataset = 'train', transform = transform_train, depth_min=args.depth_min, depth_max=args.depth_max)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
        
            from models.Depth_resnet import Depth_G
            G = Depth_G().to(args.device)
            if not args.eval:
                shutil.copy('models/Depth_resnet.py', os.path.join(args.result_path, 'G.py'))

            if args.pretrained_G is not None:
                G.load_state_dict(torch.load(last_save(args.pretrained_G, 'G*'), map_location='cpu'))
                G.to(args.device)
            G_optimizer = optim.Adam(params=G.parameters(), lr=args.G_lr)
        else:
            trainloader = testloader = G = G_optimizer = test_data = None
    
    elif args.application == 'HS':
        if args.train_G:
            from utils.dataloader.HS import CAVE, HSDB, ICVL
            transform_test = transforms.Compose([
                    transforms.CenterCrop([args.param.camera_resolution, args.param.camera_resolution]),
                    transforms.Pad(args.edge_padding, padding_mode='reflect')
                ])
            transform_train = transforms.Compose([
                    transforms.RandomCrop([args.param.camera_resolution+2*args.edge_padding, args.param.camera_resolution+2*args.edge_padding],pad_if_needed=True),
                    transforms.RandomHorizontalFlip(),
                    ChannelRolling()
                ])
            
            testset = CAVE(transform = transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
            for _, batch_data in enumerate(testloader):
                test_data = batch_data
                break
            if args.debug:
                trainloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
            else:
                trainset1 = HSDB(transform = transform_train)
                trainset2 = ICVL(transform = transform_train)
                trainset = torch.utils.data.ConcatDataset([trainset1, trainset2])
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)    

            args.param.QE_weight = args.param.QE_weight.to(args.device)   
            args.param.QE_conv_layer = torch.nn.Conv2d(in_channels=31, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
            args.param.QE_conv_layer.weight = torch.nn.Parameter(args.param.QE_weight)

            if args.PSF_file is not None:
                import pickle
                with open(args.PSF_file.replace('.npy','_approx.pkl'), 'rb') as file:
                    args.psf_approx = pickle.load(file)
            
            from models.HS_DWDN import HS_G
            G = HS_G().to(args.device)
            if not args.eval:
                shutil.copy('models/HS_DWDN.py', os.path.join(args.result_path, 'G.py'))   

            if args.pretrained_G is not None:
                G.load_state_dict(torch.load(last_save(args.pretrained_G, 'G*'), map_location='cpu'))
                G.to(args.device)
            G_optimizer = optim.Adam(params=G.parameters(), lr=args.G_lr)

        else:
            trainloader = testloader = G = G_optimizer = test_data = None  
    
    elif args.application == 'HDR':
        if args.train_G:
            from utils.dataloader.HDR import HDRi
            transform_test = transforms.Compose([
                    transforms.CenterCrop([args.param.camera_resolution+2*args.edge_padding, args.param.camera_resolution+2*args.edge_padding])
                ])
            testset = HDRi(mode = 'val', transform = transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
            test_data = testloader.dataset[1]
            test_data['image'] = test_data['image'][None,...]

            if args.debug:
                trainloader = testloader
            else:
                transform_train = transforms.Compose([
                        transforms.RandomCrop([args.param.camera_resolution+2*args.edge_padding, args.param.camera_resolution+2*args.edge_padding],pad_if_needed=True),
                        transforms.RandomHorizontalFlip()
                    ])
                trainset = HDRi(mode = 'train', clip_min = 5, clip_max = 10, transform = transform_test)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)        

            from models.HDR_DWDN import HDR_G
            G = HDR_G().to(args.device)
            if not args.eval:
                shutil.copy('models/HDR_DWDN.py', os.path.join(args.result_path, 'G.py'))   

            if args.pretrained_G is not None:
                G.load_state_dict(torch.load(last_save(args.pretrained_G, 'G*'), map_location='cpu'))
                G.to(args.device)
            G_optimizer = optim.Adam(params=G.parameters(), lr=args.G_lr)
        else:
            trainloader = testloader = G = G_optimizer = test_data = None  

    return trainloader, testloader, test_data, DOE_phase_1D, DOE_phase, optics_optimizer, G, G_optimizer
