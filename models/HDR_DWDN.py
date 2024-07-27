import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

import pado
from utils.optics_utils import *

def Wiener_deconv(image, psf):
    # todo: add noise term
    otf = pado.fft(real2complex(psf))
    wiener = otf.conj() / real2complex(otf.get_mag() **2 + 1e-4) 
    image_deconv = pado.ifft(wiener * pado.fft(real2complex(image))).get_mag()
    return image_deconv.type_as(image)

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


'''
This is based on the implementation by Kai Zhang (github: https://github.com/cszn)
'''

# --------------------------------
# --------------------------------
def get_uperleft_denominator(img, kernel):
    ker_f = convert_psf2otf(kernel, img.size()) # discrete fourier transform of kernel
    nsr = wiener_filter_para(img)
    denominator = inv_fft_kernel_est(ker_f, nsr )#
    img1 = img.cuda()
    numerator = torch.rfft(img1, 3, onesided=False)
    deblur = deconv(denominator, numerator)
    return deblur

# --------------------------------
# --------------------------------
def wiener_filter_para(_input_blur):
    median_filter = MedianPool2d(kernel_size=3, padding=1)(_input_blur)
    diff = median_filter - _input_blur
    num = (diff.shape[2]*diff.shape[2])
    mean_n = torch.sum(diff, (2,3)).view(-1,1,1,1)/num
    var_n = torch.sum((diff - mean_n) * (diff - mean_n), (2,3))/(num-1)
    mean_input = torch.sum(_input_blur, (2,3)).view(-1,1,1,1)/num
    var_s2 = (torch.sum((_input_blur-mean_input)*(_input_blur-mean_input), (2,3))/(num-1))**(0.5)
    NSR = var_n / var_s2 * 8.0 / 3.0 / 10.0
    NSR = NSR.view(-1,1,1,1)
    return NSR

# --------------------------------
# --------------------------------
def inv_fft_kernel_est(ker_f, NSR):
    inv_denominator = ker_f[:, :, :, :, 0] * ker_f[:, :, :, :, 0] \
                      + ker_f[:, :, :, :, 1] * ker_f[:, :, :, :, 1] + NSR
    # pseudo inverse kernel in flourier domain.
    inv_ker_f = torch.zeros_like(ker_f)
    inv_ker_f[:, :, :, :, 0] = ker_f[:, :, :, :, 0] / inv_denominator
    inv_ker_f[:, :, :, :, 1] = -ker_f[:, :, :, :, 1] / inv_denominator
    return inv_ker_f

# --------------------------------
# --------------------------------
def deconv(inv_ker_f, fft_input_blur):
    # delement-wise multiplication.
    deblur_f = torch.zeros_like(inv_ker_f).cuda()
    deblur_f[:, :, :, :, 0] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 0] \
                            - inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 1]
    deblur_f[:, :, :, :, 1] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 1] \
                            + inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 0]
    deblur = torch.irfft(deblur_f, 3, onesided=False)
    return deblur

# --------------------------------
# --------------------------------
def convert_psf2otf(ker, size):
    psf = torch.zeros(size).cuda()
    print(psf.shape)
    # circularly shift
    centre = ker.shape[2]//2
    psf[:, :, :centre, :centre] = ker[:, :, (centre):, (centre):]
    psf[:, :, :centre, -(centre):] = ker[:, :, (centre):, :(centre)]
    psf[:, :, -(centre):, :centre] = ker[:, :, : (centre), (centre):]
    psf[:, :, -(centre):, -(centre):] = ker[:, :, :(centre), :(centre)]
    # compute the otf
    otf = torch.rfft(psf, 3, onesided=False)
    return otf


def postprocess(*images, rgb_range):
    def _postprocess(img):
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    return [_postprocess(img) for img in images]

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class Conv(nn.Module):
    def __init__(self , input_channels , n_feats , kernel_size , stride = 1 ,padding=0 , bias=True , bn = False , act=False ):
        super(Conv , self).__init__()
        m = []
        m.append(nn.Conv2d(input_channels , n_feats , kernel_size , stride , padding , bias=bias))
        if bn: m.append(nn.BatchNorm2d(n_feats))
        if act:m.append(nn.ReLU())
        self.body = nn.Sequential(*m)
    def forward(self, input):
        return self.body(input)

class Deconv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=2, padding=0, output_padding=0 , bias=True, act=False):
        super(Deconv, self).__init__()
        m = []
        m.append(nn.ConvTranspose2d(input_channels, n_feats, kernel_size, stride=stride, padding=padding,output_padding=output_padding, bias=bias))
        if act: m.append(nn.ReLU())
        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)

class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, padding = 0 ,bias=True, bn=False, act=nn.ReLU(), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, padding = padding , bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        out = res.clone() + x

        return out

class HDR_G(nn.Module):
    def __init__(self):
        super().__init__()

        n_resblock = 3
        n_feats1 = 6
        n_feats = 32
        kernel_size = 5
        self.n_colors = 3

        FeatureBlock = [Conv(self.n_colors, n_feats1, kernel_size, padding=2, act=True),
                        ResBlock(Conv, n_feats1, kernel_size, padding=2),
                        ResBlock(Conv, n_feats1, kernel_size, padding=2),
                        ResBlock(Conv, n_feats1, kernel_size, padding=2)]

        InBlock1 = [Conv(3*n_feats1, n_feats, kernel_size, padding=2, act=True),
                   ResBlock(Conv, n_feats, kernel_size, padding=2),
                   ResBlock(Conv, n_feats, kernel_size, padding=2),
                   ResBlock(Conv, n_feats, kernel_size, padding=2)]
        InBlock2 = [Conv(3*n_feats1 + n_feats, n_feats, kernel_size, padding=2, act=True),
                   ResBlock(Conv, n_feats, kernel_size, padding=2),
                   ResBlock(Conv, n_feats, kernel_size, padding=2),
                   ResBlock(Conv, n_feats, kernel_size, padding=2)]

        # encoder1
        Encoder_first= [Conv(n_feats , n_feats*2 , kernel_size , padding = 2 ,stride=2 , act=True),
                        ResBlock(Conv , n_feats*2 , kernel_size ,padding=2),
                        ResBlock(Conv , n_feats*2 , kernel_size ,padding=2),
                        ResBlock(Conv , n_feats*2 , kernel_size ,padding=2)]
        # encoder2
        Encoder_second = [Conv(n_feats*2 , n_feats*4 , kernel_size , padding=2 , stride=2 , act=True),
                          ResBlock(Conv , n_feats*4 , kernel_size , padding=2),
                          ResBlock(Conv , n_feats*4 , kernel_size , padding=2),
                          ResBlock(Conv , n_feats*4 , kernel_size , padding=2)]
        # decoder2
        Decoder_second = [ResBlock(Conv , n_feats*4 , kernel_size , padding=2) for _ in range(n_resblock)]
        Decoder_second.append(Deconv(n_feats*4 , n_feats*2 ,kernel_size=3 , padding=1 , output_padding=1 , act=True))
        # decoder1
        Decoder_first = [ResBlock(Conv , n_feats*2 , kernel_size , padding=2) for _ in range(n_resblock)]
        Decoder_first.append(Deconv(n_feats*2 , n_feats , kernel_size=3 , padding=1, output_padding=1 , act=True))

        OutBlock = [ResBlock(Conv , n_feats , kernel_size , padding=2) for _ in range(n_resblock)]

        OutBlock2 = [Conv(n_feats + 3, self.n_colors, kernel_size , padding=2)]

        self.FeatureBlock = nn.Sequential(*FeatureBlock)
        self.inBlock1 = nn.Sequential(*InBlock1)
        self.inBlock2 = nn.Sequential(*InBlock2)
        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.outBlock = nn.Sequential(*OutBlock)
        self.outBlock2 = nn.Sequential(*OutBlock2)

    def forward(self, capture_enc, capture_ref, kernel):
        kernel = torch.tile(kernel, (1,2,1,1))
        capture_ldr = []
            
        enc_feature_out = self.FeatureBlock(capture_enc)
        ref_feature_out = self.FeatureBlock(capture_ref)
        enc_clear_features = Wiener_deconv(enc_feature_out, kernel)

        input = torch.cat([enc_feature_out, ref_feature_out, enc_clear_features], 1)
        self.n_levels = 2
        self.scale = 0.5

        for level in range(self.n_levels):
            scale = self.scale ** (self.n_levels - level - 1)
            n, c, h, w = input.shape
            hi = int(round(h * scale))
            wi = int(round(w * scale))
            if level == 0:
                input_clear = F.interpolate(input, (hi, wi), mode='bilinear')
                inp_all = input_clear
                first_scale_inblock = self.inBlock1(inp_all)
            else:
                input_clear = F.interpolate(input, (hi, wi), mode='bilinear')
                input_pred = F.interpolate(input_pre, (hi, wi), mode='bilinear')
                inp_all = torch.cat((input_clear, input_pred), 1)
                first_scale_inblock = self.inBlock2(inp_all)
            if hi == h:
                capture_ldr = capture_ref * 2
            else:
                capture_ldr = F.interpolate(capture_ref * 2, (hi, wi), mode='bilinear')

            first_scale_encoder_first = self.encoder_first(first_scale_inblock)
            first_scale_encoder_second = self.encoder_second(first_scale_encoder_first)
            first_scale_decoder_second = self.decoder_second(first_scale_encoder_second)
            first_scale_decoder_first = self.decoder_first(first_scale_decoder_second+first_scale_encoder_first)
            input_pre = self.outBlock(first_scale_decoder_first+first_scale_inblock)
            out = self.outBlock2(torch.cat([capture_ldr,input_pre],1))

        return out



