import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

def disable_tracking():
    if hasattr(torch, "_C"):
        if hasattr(torch._C, "_jit_set_profiling_executor"):
            torch._C._jit_set_profiling_executor(False)
        if hasattr(torch._C, "_jit_set_profiling_mode"):
            torch._C._jit_set_profiling_mode(False)
    return True

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.TCMamba import TripletAttention
from models.wavelet_modules import DynamicWaveletBlock, VascularWaveletUp, VascularWaveletDown
from models.fusion_modules import HybridFusionModule


class WMDiff(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, base_ch=64, growth_rate=32):
        super().__init__()
        disable_tracking()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            DynamicWaveletBlock(base_ch, growth_rate),
            nn.Conv2d(base_ch, base_ch, 1)
        )
        self.pool1 = VascularWaveletDown(in_ch=base_ch, out_ch=base_ch)

        self.enc2 = nn.Sequential(
            DynamicWaveletBlock(base_ch, growth_rate),
            nn.Conv2d(base_ch, base_ch * 2, 1)
        )
        self.pool2 = VascularWaveletDown(in_ch=base_ch * 2, out_ch=base_ch * 2)

        self.enc3 = nn.Sequential(
            DynamicWaveletBlock(base_ch * 2, growth_rate),
            nn.Conv2d(base_ch * 2, base_ch * 4, 1)
        )
        self.pool3 = VascularWaveletDown(in_ch=base_ch * 4, out_ch=base_ch * 4)

        self.bottleneck = nn.Sequential(
            TripletAttention(channels=base_ch * 4),
            nn.Conv2d(base_ch * 4, base_ch * 8, 1)
        )
        self.up3 = VascularWaveletUp(in_ch=base_ch * 8, out_ch=base_ch * 4)
        self.fusion3 = HybridFusionModule(
            in_channels_list=[base_ch * 4, base_ch * 2, base_ch],
            out_channels=base_ch * 4
        )
        self.dec3 = nn.Sequential(
            DynamicWaveletBlock(base_ch * 4 * 3, growth_rate),
            nn.Conv2d(base_ch * 4 * 3, base_ch * 4, 1)
        )

        self.up2 = VascularWaveletUp(in_ch=base_ch * 4, out_ch=base_ch * 2)
        self.fusion2 = HybridFusionModule(
            in_channels_list=[base_ch * 2, base_ch, base_ch * 2],
            out_channels=base_ch * 2
        )
        self.dec2 = nn.Sequential(
            DynamicWaveletBlock(base_ch * 2 * 3, growth_rate),
            nn.Conv2d(base_ch * 2 * 3, base_ch * 2, 1)
        )

        self.up1 = VascularWaveletUp(in_ch=base_ch * 2, out_ch=base_ch)
        self.fusion1 = HybridFusionModule(
            in_channels_list=[base_ch, base_ch, base_ch],
            out_channels=base_ch
        )
        self.dec1 = nn.Sequential(
            DynamicWaveletBlock(base_ch * 2, growth_rate),
            nn.Conv2d(base_ch * 2, base_ch, 1)
        )

        out_channels = max(4, ((out_channels + 3) // 4) * 4)
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_ch, out_channels, 1, groups=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if out_channels != 2:
            self.final_adjust = nn.Conv2d(out_channels, 2, 1)
        else:
            self.final_adjust = nn.Identity()

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            gain = nn.init.calculate_gain('relu', param=0.2)
            std = gain / math.sqrt(fan_in)
            nn.init.normal_(module.weight, 0, std)

        elif isinstance(module, nn.ConvTranspose2d):
            fan_in = module.in_channels
            gain = nn.init.calculate_gain('relu')
            std = gain / math.sqrt(fan_in)
            nn.init.normal_(module.weight, 0, std)

        if isinstance(module, nn.Conv2d) and module.__class__.__name__ == 'Conv2d' and hasattr(self, 'final_conv'):
            if module in self.final_conv.modules():
                fan_in = module.in_channels // module.groups
                gain = nn.init.calculate_gain('relu')
                std = gain / math.sqrt(fan_in)
                nn.init.normal_(module.weight, 0, std)

    def initialize(self):
        self.apply(self._init_weights)
        return self

    def forward(self, x, timesteps=None):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        bottle = self.bottleneck(p3)

        d3_up = self.up3(bottle)
        e1_avg = F.adaptive_avg_pool2d(e1, d3_up.shape[2:])
        e2_avg = F.adaptive_avg_pool2d(e2, d3_up.shape[2:])
        f3 = self.fusion3(e3, e2_avg, e1_avg)
        d3 = torch.cat([f3, e3, d3_up], dim=1)
        d3 = self.dec3(d3)

        d2_up = self.up2(d3)
        e1_avg_2 = F.adaptive_avg_pool2d(e1, d2_up.shape[2:])
        e3_up = self.up2(e3)
        f2 = self.fusion2(e2, e1_avg_2, e3_up)
        d2 = torch.cat([f2, e2, d2_up], dim=1)
        d2 = self.dec2(d2)
        d1_up = self.up1(d2)
        e2_up = self.up1(e2)
        e3_up = self.up2(e3)
        e3_up = self.up1(e3_up)
        f1 = self.fusion1(e1, e2_up, e3_up)
        d1 = torch.cat([f1, d1_up], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        out = self.final_adjust(out)
        return F.softmax(out, dim=1)


if __name__ == "__main__":
    net = WMDiff(in_channels=3, out_channels=2)
    x = torch.randn(2, 3, 256, 256)
    timesteps = torch.zeros(2, dtype=torch.long)
    print("Output shape:", net(x, timesteps).shape)

    print("Output shape (without timesteps):", net(x).shape)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
