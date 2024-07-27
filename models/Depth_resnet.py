import torch
from torch import nn
import torchvision.models as models

class Depth_G(nn.Module):
    def __init__(self, nf=16):
        super().__init__()

        # ResNet-34 encoder
        model = models.resnet18(pretrained=False)
        modules = list(model.children())[:-2]
        self.resnet_enc = nn.Sequential(*modules)
        self.resnet_ref = nn.Sequential(*modules)
        
        for param in self.resnet_enc.parameters():
            param.requires_grad = True

        for param in self.resnet_ref.parameters():
            param.requires_grad = True

        # Upsampling blocks
        self.up5_0 = self.ConvTranspBlock(1024, 512, 2, 2, 0)
        self.up5_1 = self.DoubleConvBlock(512, 512, 3, 1, 1)
        
        self.up4_0 = self.ConvTranspBlock(512, 256, 2, 2, 0)
        self.up4_1 = self.DoubleConvBlock(256, 256, 3, 1, 1)

        self.up3_0 = self.ConvTranspBlock(256, 128, 2, 2, 0)
        self.up3_1 = self.DoubleConvBlock(128, 128, 3, 1, 1)

        self.up2_0 = self.ConvTranspBlock(128, 64, 2, 2, 0)
        self.up2_1 = self.DoubleConvBlock(64, 64, 3, 1, 1)

        self.up1_0 = self.ConvTranspBlock(64, 32, 2, 2, 0)
        self.up1_1 = self.DoubleConvBlock(32, 32, 3, 1, 1)

        # Final convolution to get the depth map
        self.final_conv = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, capture_enc, capture_ref, kernel):
        # Feature extraction
        res_features_enc = self.resnet_enc(capture_enc)
        res_features_ref = self.resnet_ref(capture_ref)
        res_features = torch.cat([res_features_enc, res_features_ref], dim=1) # B, 1024, 20, 20

        # Upsampling with Up-Convolution Blocks
        up5 = self.up5_0(res_features)
        up5 = self.up5_1(up5)

        up4 = self.up4_0(up5)
        up4 = self.up4_1(up4)

        up3 = self.up3_0(up4)
        up3 = self.up3_1(up3)

        up2 = self.up2_0(up3)
        up2 = self.up2_1(up2)

        up1 = self.up1_0(up2)
        up1 = self.up1_1(up1)

        # Final depth map
        out_depth = self.final_conv(up1)

        return out_depth

    def UpBlock(self, in_channels, out_channels, scale_factor):
        block = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def ConvBlock(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
                    nn.ReflectionPad2d(padding),
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(),
                )
        return block
    
    def DoubleConvBlock(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
                    nn.ReflectionPad2d(padding),
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(),
                    nn.ReflectionPad2d(padding),
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                )
        return block

    def ConvTranspBlock(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
                    nn.ReflectionPad2d(padding),
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(),
                )
        return block