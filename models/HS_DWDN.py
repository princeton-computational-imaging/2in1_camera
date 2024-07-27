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

class HS_G(nn.Module):
    def __init__(self):
        super().__init__()

        n_resblock = 3
        n_feats1 = 6
        n_feats = 64
        kernel_size = 5
        self.n_colors = 31

        FeatureBlock_enc = [Conv(3, self.n_colors, kernel_size, padding=2, act=True),
                        ResBlock(Conv, self.n_colors, kernel_size, padding=2),
                        ResBlock(Conv, self.n_colors, kernel_size, padding=2),
                        ResBlock(Conv, self.n_colors, kernel_size, padding=2)]

        FeatureBlock_ref = [Conv(3, n_feats1, kernel_size, padding=2, act=True),
                        ResBlock(Conv, n_feats1, kernel_size, padding=2),
                        ResBlock(Conv, n_feats1, kernel_size, padding=2),
                        ResBlock(Conv, n_feats1, kernel_size, padding=2)]

        InBlock1 = [Conv(self.n_colors + 2 * n_feats1, n_feats, kernel_size, padding=2, act=True),
                   ResBlock(Conv, n_feats, kernel_size, padding=2),
                   ResBlock(Conv, n_feats, kernel_size, padding=2),
                   ResBlock(Conv, n_feats, kernel_size, padding=2)]
        InBlock2 = [Conv(self.n_colors + 2 * n_feats1 + n_feats, n_feats, kernel_size, padding=2, act=True),
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

        OutBlock2 = [Conv(n_feats , self.n_colors, kernel_size , padding=2)]

        self.FeatureBlock_enc = nn.Sequential(*FeatureBlock_enc)
        self.FeatureBlock_ref = nn.Sequential(*FeatureBlock_ref)
        self.inBlock1 = nn.Sequential(*InBlock1)
        self.inBlock2 = nn.Sequential(*InBlock2)
        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.outBlock = nn.Sequential(*OutBlock)
        self.outBlock2 = nn.Sequential(*OutBlock2)

    def forward(self, capture_enc, capture_ref, kernel):

        enc_feature_out_hs = self.FeatureBlock_enc(capture_enc)
        enc_feature_out = self.FeatureBlock_ref(capture_enc)
        ref_feature_out = self.FeatureBlock_ref(capture_ref)
        enc_clear_features = Wiener_deconv(enc_feature_out_hs, kernel)

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

            first_scale_encoder_first = self.encoder_first(first_scale_inblock)
            first_scale_encoder_second = self.encoder_second(first_scale_encoder_first)
            first_scale_decoder_second = self.decoder_second(first_scale_encoder_second)
            first_scale_decoder_first = self.decoder_first(first_scale_decoder_second+first_scale_encoder_first)
            input_pre = self.outBlock(first_scale_decoder_first+first_scale_inblock)
            out = self.outBlock2(input_pre)

        return out



