import numpy as np
import torch
import pado

def max_diffraction_pxl(wvl, DOE_pitch, focal_length, camera_pitch):
    return np.tan(np.sinh(wvl/(2*DOE_pitch)))*focal_length / camera_pitch 

def Fresnel_number(wvl, focal_length, aperture):
    return ((aperture/2)**2)/(focal_length * wvl)

def circle_mask(R,cutoff):
    [x, y] = np.mgrid[-int(R):int(R),-int(R):int(R)]
    dist = np.sqrt(x**2 +y**2).astype(np.int32)
    mask = torch.tensor(1.0*(dist < cutoff))[None, None, ...]
    return mask

def DOE_1Dto2D(DOE_1D, args):
    param = args.param
    [x, y] = np.mgrid[-int(param.R/2):int(param.R/2),-int(param.R/2):int(param.R/2)]
    dist = np.sqrt(x**2 +y**2).astype(np.int32)
    DOE_2D = DOE_1D[dist.reshape((-1,))].reshape((param.R,param.R))[None,None,...]
    return DOE_2D.to(args.device)

def sample_psf(psf, sample_ratio):
    if sample_ratio == 1:
        return psf
    else:
        return torch.nn.AvgPool2d(sample_ratio, stride=sample_ratio)(psf)

def real2complex(real):
    return pado.Complex(mag=real, ang=torch.zeros_like(real))

def compute_pad_size(current_size, target_size):
    assert current_size < target_size
    gap = target_size - current_size
    left = int(gap/2)
    right = gap - left
    return int(left), int(right)

def sin_grating(sf, ori, phase, R):
    """
    :param sf: spatial frequency (in pixels)
    :param ori: wave orientation (in degrees, [0-360])
    :param phase: wave phase (in degrees, [0-360])
    :param R: resolution (integer)
    :return: numpy array of shape (R, R)
    """
    # Get x and y coordinates
    x, y = np.meshgrid(np.arange(R), np.arange(R))

    # Get the appropriate gradient
    gradient = np.sin(ori * np.pi / 180) * x - np.cos(ori * np.pi / 180) * y

    # Plug gradient into wave function
    grating = np.sin((2 * np.pi * gradient) / sf + (phase * np.pi) / 180)
    
    return torch.Tensor(grating)

def AFoV(focal, sensor_dim):
    AFoV = 2*np.arctan(sensor_dim/(2*focal))
    return AFoV*180

# aperture split
def half_aperture_mask(width_one, width_zero, R):
    half_one = torch.ones([1,1, R,width_one])
    half_zero = torch.ones([1,1,R,width_zero]) * 1e-20
    return torch.cat([half_one, half_zero], dim = -1),torch.cat([half_zero, half_one], dim = -1)

