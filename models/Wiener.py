import numpy as np
import torch
import torch.nn as nn
import pado

from utils.optics_utils import *

class Arch(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, image, psf_far):

        image_deconv = Wiener_deconv(image, psf_far)
        return image_deconv

def Wiener_deconv(image, psf):
    # todo: add noise term
    otf = pado.fft(real2complex(psf))
    wiener = otf.conj() / real2complex(otf.get_mag() **2 + 1e-4) 
    image_deconv = pado.ifft(wiener * pado.fft(real2complex(image))).get_mag()
    return torch.clamp(image_deconv.type_as(image), min = 0, max = 1)
