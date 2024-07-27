import torch
import numpy as np
import scipy
from .fourier import fft, ifft, fftshift, ifftshift
from .complex import Complex
from .conv import conv_fft


def compute_pad_width(field, linear):
    """
    Compute the pad width of an array for FFT-based convolution
    Args:
        field: (B,Ch,R,C) complex tensor
        linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
    Returns:
        pad_width: pad-width tensor
    """

    if linear:
        R,C = field.shape()[-2:]
        pad_width = (C//2, C//2, R//2, R//2)
    else:
        pad_width = (0,0,0,0)
    return pad_width 

def unpad(field_padded, pad_width):
    """
    Unpad the already-padded complex tensor 
    Args:
        field_padded: (B,Ch,R,C) padded complex tensor 
        pad_width: pad-width tensor
    Returns:
        field: unpadded complex tensor
    """

    field = field_padded[...,pad_width[2]:-pad_width[3],pad_width[0]:-pad_width[1]]
    return field

def fresnel_number(wid, z, wvl):
    return (wid * wid) / (z * wvl)

def critical_sampling(z, wvl, length):
    return wvl * z / length

class Propagator:
    def __init__(self, mode):
        """
        Free-space propagator of light waves
        One can simulate the propagation of light waves on free space (no medium change at all).
        Args:
            mode: type of propagator. currently, we support "Fraunhofer" propagation or "Fresnel" propagation. Use Fraunhofer for far-field propagation and Fresnel for near-field propagation. 
        """
        self.mode = mode

    def forward(self, light, z, linear=True):
        """
        Forward the incident light with the propagator. 
        Args:
            light: incident light 
            z: propagation distance in meter
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        """

        if self.mode == 'auto':
            f_num = fresnel_number(light.R*light.pitch/2, z, light.wvl)
            if f_num > 5:
                return self.forward_asm(light, z)
            elif f_num < 0.2:
                return self.forward_Fraunhofer(light, z, linear)
            else:
                return self.forward_Fresnel(light, z, linear)

        elif self.mode == 'Fraunhofer':
            return self.forward_Fraunhofer(light, z, linear)
        elif self.mode == 'Fresnel':
            return self.forward_Fresnel(light, z, linear)
        elif self.mode == 'ASM':
            return self.forward_asm(light, z)
        else:
            return NotImplementedError('%s propagator is not implemented'%self.mode)


    def forward_Fraunhofer(self, light, z, linear=True):
        """
        Forward the incident light with the Fraunhofer propagator. 
        Args:
            light: incident light 
            z: propagation distance in meter. 
                The propagated wavefront is independent w.r.t. the travel distance z.
                The distance z only affects the size of the "pixel", effectively adjusting the entire image size.
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        """

        # pad_width = compute_pad_width(light.field, linear)
        field_propagated = fft(light.field) #, pad_width=pad_width)
        # field_propagated = unpad(field_propagated, pad_width)

        # based on the Fraunhofer reparametrization (u=x/wvl*z) and the Fourier frequency sampling (1/bandwidth)
        bw_r = light.get_bandwidth()[0]
        bw_c = light.get_bandwidth()[1]
        pitch_r_after_propagation = light.wvl*z/bw_r
        pitch_c_after_propagation = light.wvl*z/bw_c

        light_propagated = light.clone()

        # match the x-y pixel pitch using resampling
        if pitch_r_after_propagation >= pitch_c_after_propagation:
            scale_c = 1
            scale_r = pitch_r_after_propagation/pitch_c_after_propagation
            pitch_after_propagation = pitch_c_after_propagation
        elif pitch_r_after_propagation < pitch_c_after_propagation:
            scale_r = 1
            scale_c = pitch_c_after_propagation/pitch_r_after_propagation
            pitch_after_propagation = pitch_r_after_propagation

        field_propagated.to_polar()
        light_propagated.set_field(field_propagated)
        light_propagated.magnify((scale_r,scale_c))
        light_propagated.set_pitch(pitch_after_propagation)

        return light_propagated


    def forward_Fresnel(self, light, z, linear):
        """
        Forward the incident light with the Fresnel propagator. 
        Args:
            light: incident light 
            z: propagation distance in meter. 
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        """
        field_input = light.field

        # compute the convolutional kernel
        sx = light.C / 2
        sy = light.R / 2
        x = np.arange(-sx, sx, 1)
        y = np.arange(-sy, sy, 1)
        xx, yy = np.meshgrid(x,y)
        xx = torch.from_numpy(xx*light.pitch).to(light.device)
        yy = torch.from_numpy(yy*light.pitch).to(light.device)
        k = 2*np.pi/light.wvl  # wavenumber
        phase = (k*(xx**2 + yy**2)/(2*z))
        amplitude = torch.ones_like(phase) / z / light.wvl
        conv_kernel = Complex(mag=amplitude, ang=phase) 
        
        # Propagation with the convolution kernel
        pad_width = compute_pad_width(field_input, linear)
        
        field_propagated = conv_fft(field_input, conv_kernel, pad_width)

        # return the propagated light
        light_propagated = light.clone()
        light_propagated.set_field(field_propagated)

        return light_propagated

    def forward_asm(self, light, z):
        """
        Forward the incident light with the ASM propagator.
        Args:
            light: incident light
            z: propagation distance in meter.
        Returns:
            light: light after propagation
        """

        field_input = light.field
        fft_c = fft(field_input)
        c = fftshift(fft_c)

        fx = np.fft.fftfreq(light.R, d=light.pitch)
        fy = np.fft.fftfreq(light.C, d=light.pitch)
        fxx, fyy = np.meshgrid(fx, fy)

        arg = (2*np.pi)**2 * ((1. / light.wvl) ** 2 - fxx ** 2 - fyy ** 2)

        tmp = np.sqrt(np.abs(arg))
        kz = np.where(arg >= 0, tmp, 1j*tmp)

        c.to_native()
        c.native = c.native * torch.from_numpy(np.exp(1j*kz*z)).to(light.device)
        c.to_polar()
        field_propagated = ifft(ifftshift(c))

        light_propagated = light.clone()
        light_propagated.set_field(field_propagated)

        return light_propagated
