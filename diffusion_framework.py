import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import sys
from torch.utils.checkpoint import checkpoint


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device='cuda'):
    noise = torch.rand_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def forward_diffusion_sample_last(x_0, t, device='cuda'):
    noise = (torch.randn_like(x_0)+1)/2
    t_minusone = t.detach().cpu().numpy()-1
    t_minusone[t_minusone==-1] = 0
    t_minusone = torch.tensor(t_minusone).to(device)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_alphas_cumprod_t_minusone = get_index_from_list(sqrt_alphas_cumprod, t_minusone, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    sqrt_one_minus_alphas_cumprod_t_minusone = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t_minusone, x_0.shape
    )
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), sqrt_alphas_cumprod_t_minusone.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t_minusone.to(device) * noise.to(device)


T = 100
betas = linear_beta_schedule(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.WMDiff import WMDiff
from models.wavelet_modules import DynamicWaveletBlock
from models.attention_modules import LiteWindowAttention, LiteSKTransformer, ChannelGate

class DeformableAttention(nn.Module):
    def __init__(self, dim, num_heads=4, groups=4, num_points=4):
        super().__init__()
        self.groups = min(groups, dim)
        while dim % self.groups != 0:
            self.groups -= 1
        
        assert dim % self.groups == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.num_points = num_points
        self.num_heads = min(num_heads, dim // 4)
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.offset_conv = nn.Sequential(
            nn.Conv2d(dim, dim // self.groups, 3, padding=1, groups=self.groups),
            nn.GELU(),
            nn.Conv2d(dim // self.groups, 3 * self.num_heads * num_points, 1)
        )

        self.q_proj = nn.Conv2d(dim, dim, 1, groups=self.groups)
        self.k_proj = nn.Conv2d(dim, dim, 1, groups=self.groups)
        self.v_proj = nn.Conv2d(dim, dim, 1, groups=self.groups)
        self.proj = nn.Conv2d(dim, dim, 1, groups=self.groups)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x

        offset_mask = self.offset_conv(x)
        offset = offset_mask[:, :2 * self.num_heads * self.num_points] * 0.1
        mask = offset_mask[:, 2 * self.num_heads * self.num_points:]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn = q.view(B, self.num_heads, -1, H*W) 
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.einsum('bhcn,bhcn->bhcn', attn, v.view(B, self.num_heads, -1, H*W))
        out = out.view(B, C, H, W)

        return self.proj(out) + shortcut

class EfficientMultiScaleTransformer(nn.Module):
    def __init__(self, ch_list, out_channels, num_heads=4, groups=4):
        super().__init__()
        self.hidden_dim = max(out_channels // 2, 16)
        groups = min(groups, self.hidden_dim)
        while self.hidden_dim % groups != 0:
            groups -= 1
            
        self.scale_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, self.hidden_dim, 1, groups=1),
                nn.GELU()
            ) for ch in ch_list
        ])
        
        self.attention = DeformableAttention(self.hidden_dim, num_heads, groups)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(len(ch_list) * self.hidden_dim, out_channels, 1, groups=1),
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self, x1, x2, x4):
        target_size = x1.shape[2:]
        features = []
        for i, x in enumerate([x1, x2, x4]):
            x = F.interpolate(x, target_size, mode='bilinear', align_corners=True)
            x = self.scale_adapters[i](x)
            x = self.attention(x)
            features.append(x)
        return self.fusion(torch.cat(features, dim=1))

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context):
        b, c, h, w = x.shape
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q = q.view(b, self.heads, -1, h * w)
        k = k.view(b, self.heads, -1, h * w)
        v = v.view(b, self.heads, -1, h * w)
        
        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.view(b, -1, h, w)
        
        return self.to_out(out)

class DiffusionBlock(nn.Module):
    def __init__(self, dim, dim_out=None, *, cond_dim=None, groups=8):
        super().__init__()
        dim_out = dim_out or dim
        self.dim = dim
        self.dim_out = dim_out
        
        self.groups = min(groups, dim)
        while dim % self.groups != 0:
            self.groups -= 1
            if self.groups <= 1:
                self.groups = 1
                break
                
        self.groups_out = min(groups, dim_out)
        while dim_out % self.groups_out != 0:
            self.groups_out -= 1
            if self.groups_out <= 1:
                self.groups_out = 1
                break
        
        self.channel_adapter = None
        
        self.norm1 = nn.InstanceNorm2d(dim)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(dim, dim_out, 3, padding=1)
        
        self.norm2 = nn.InstanceNorm2d(dim_out)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        
        self.has_cond = cond_dim is not None
        if self.has_cond:
            self.cond_proj = nn.Sequential(
                nn.Conv2d(cond_dim, dim_out, 1),
                nn.SiLU()
            )
            self.cross_attn = CrossAttention(dim_out, heads=8, dim_head=64)
            self.cond_gate = nn.Sequential(
                nn.InstanceNorm2d(dim_out),
                nn.Conv2d(dim_out, dim_out, 1),
                nn.Sigmoid()
            )
            self.pos_embedding = nn.Sequential(
                nn.Conv2d(dim_out, dim_out, 3, padding=1, groups=groups),
                nn.GELU(),
                nn.Conv2d(dim_out, dim_out, 1)
            )
    
    def forward(self, x, cond=None):
        B, C, H, W = x.shape
        if C != self.dim:
            if self.channel_adapter is None or self.channel_adapter.in_channels != C:
                self.channel_adapter = nn.Conv2d(C, self.dim, 1).to(x.device)
                nn.init.normal_(self.channel_adapter.weight, std=0.02)
                nn.init.zeros_(self.channel_adapter.bias)
            x = self.channel_adapter(x)
        
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        if self.has_cond and cond is not None:
            B, C, H, W = h.shape
            cond_hw = cond.shape[2:]
            
            if cond_hw != (H, W):
                cond = F.interpolate(cond, size=(H, W), mode='bilinear', align_corners=True)
            
            if cond.shape[1] != self.dim_out:
                cond = self.cond_proj(cond)
            
            pos_emb = self.pos_embedding(h)
            h = h + pos_emb
            cond = cond + pos_emb
            
            h_cond = self.cross_attn(h, cond)
            
            gate = self.cond_gate(h_cond)
            h = h * gate + h_cond * (1 - gate)
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        
        res_x = x if self.channel_adapter is None else x
        return h + self.res_conv(res_x)

class UpsampleBlock(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class DownsampleBlock(nn.Module):

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1, stride=2)

    def forward(self, x):
        return self.conv(x)

class DiffusionCondUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=2,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        base_ch=32,
        growth_rate=32,
        use_checkpoint=False
    ):
        super().__init__()

        dims = [dim * m for m in dim_mults]
        self.init_conv = nn.Conv2d(in_channels, dim, 3, padding=1)
        in_out = list(zip(dims[:-1], dims[1:]))
        self.cond_net = WMDiff(in_channels=in_channels, out_channels=out_channels, base_ch=base_ch, growth_rate=growth_rate)
        self.use_checkpoint = use_checkpoint

        self.down_blocks = nn.ModuleList([])
        curr_dim = dim
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.down_blocks.append(nn.ModuleList([
                DiffusionBlock(curr_dim, dim_in),
                DiffusionBlock(dim_in, dim_in, cond_dim=base_ch * (2**ind)),
                DownsampleBlock(dim_in, dim_out)
            ]))
            curr_dim = dim_out

        self.mid_block1 = DiffusionBlock(curr_dim, curr_dim)
        self.mid_attn = DiffusionBlock(curr_dim, curr_dim, cond_dim=base_ch * 8)
        self.mid_block2 = DiffusionBlock(curr_dim, curr_dim)

        self.up_blocks = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.up_blocks.append(nn.ModuleList([
                DiffusionBlock(dim_out + dim_in, dim_out),
                DiffusionBlock(dim_out, dim_out, cond_dim=base_ch * (2**(len(in_out)-ind-1))),
                UpsampleBlock(dim_out, dim_in)
            ]))
        self.final_conv = nn.Sequential(
            DiffusionBlock(dim, dim),
            nn.Conv2d(dim, out_channels, 1)
        )

    def forward(self, x, timesteps=None, x_original=None, train_cond_net=False):
        cond_input = x_original if x_original is not None else x
        
        cond_context = torch.enable_grad() if train_cond_net else torch.no_grad()
        
        with cond_context:
            if not train_cond_net:
                self.cond_net.eval()
            else:
                self.cond_net.train()

            cond_features = []

            def save_features_hook(module, input, output):
                # Detach features to reduce memory footprint when gradients are not required
                if not train_cond_net:
                    cond_features.append(output.detach())
                else:
                    cond_features.append(output)

            handles = []
            handles.append(self.cond_net.enc1.register_forward_hook(save_features_hook))
            handles.append(self.cond_net.enc2.register_forward_hook(save_features_hook))
            handles.append(self.cond_net.enc3.register_forward_hook(save_features_hook))
            handles.append(self.cond_net.bottleneck.register_forward_hook(save_features_hook))
            handles.append(self.cond_net.dec3.register_forward_hook(save_features_hook))
            handles.append(self.cond_net.dec2.register_forward_hook(save_features_hook))
            handles.append(self.cond_net.dec1.register_forward_hook(save_features_hook))

            _ = self.cond_net(cond_input)

            for handle in handles:
                handle.remove()

        x = self.init_conv(x)
        h = [x]

        for i, (block1, block2, downsample) in enumerate(self.down_blocks):
            if self.use_checkpoint and train_cond_net and self.training:
                x = checkpoint(block1, x, use_reentrant=False)
            else:
                x = block1(x)
            cond_feat = cond_features[i].detach() if not train_cond_net else cond_features[i]
            if self.use_checkpoint and train_cond_net and self.training:
                x = checkpoint(block2, x, cond_feat, use_reentrant=False)
            else:
                x = block2(x, cond_feat)
            x = downsample(x)
            h.append(x)

        x = self.mid_block1(x)
        cond_feat_mid = cond_features[3].detach() if not train_cond_net else cond_features[3]
        x = self.mid_attn(x, cond_feat_mid)
        x = self.mid_block2(x)

        for i, (block1, block2, upsample) in enumerate(self.up_blocks):
            skip_conn = h.pop()
            x = torch.cat([x, skip_conn], dim=1)
            x = block1(x)
            cond_feat = cond_features[4 + i].detach() if not train_cond_net else cond_features[4 + i]
            x = block2(x, cond_feat)
            x = upsample(x)
        
        final_skip = h.pop()
        x = torch.cat([x, final_skip], dim=1)
        x = self.final_conv(x)

        del cond_features
        if x.device.type == 'cuda':
            torch.cuda.empty_cache()

        return x

class HybridFusionModule(nn.Module):
    def __init__(self, in_channels_list, out_channels, window_ratios=[0.125, 0.0625], deform_groups=4):
        super().__init__()
        total_in_channels = sum(in_channels_list)
        
        sk_channel = max(total_in_channels, 32)
        if total_in_channels < sk_channel:
            self.channel_adapter = nn.Conv2d(total_in_channels, sk_channel, 1)
            use_sk_channel = sk_channel
        else:
            self.channel_adapter = nn.Identity()
            use_sk_channel = total_in_channels
            
        adapted_window_ratios = [min(ratio, 0.0625) for ratio in window_ratios]
        
        self.sk_selector = LiteSKTransformer(
            channel=use_sk_channel,
            reductions=8,
            num_heads=4,
            window_ratios=adapted_window_ratios
        )

        self.adapter = nn.Sequential(
            nn.Conv2d(use_sk_channel, out_channels // 2, 1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU()
        )

        self.deform_attn = EfficientMultiScaleTransformer(
            ch_list=[out_channels // 2] * 3,
            out_channels=out_channels // 2,
            groups=min(deform_groups, out_channels // 2)
        )

        self.final_fusion = nn.Sequential(
            nn.Conv2d(out_channels // 2 * 3, out_channels, 1),
            ChannelGate(out_channels)
        )

        self.res_adapter = nn.Sequential(
            nn.Conv2d(in_channels_list[0], out_channels, 1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels_list[0] != out_channels else nn.Identity()

    def forward(self, x1, x2, x4):
        target_size = x1.shape[2:]

        sk_input = torch.cat([
            x1,
            F.interpolate(x2, target_size, mode='bilinear'),
            F.interpolate(x4, target_size, mode='bilinear')
        ], dim=1)
        
        sk_input = self.channel_adapter(sk_input)
        adapted = self.adapter(self.sk_selector(sk_input))

        x2_ada = F.adaptive_avg_pool2d(adapted, (target_size[0] // 2, target_size[1] // 2))
        x4_ada = F.adaptive_avg_pool2d(adapted, (target_size[0] // 4, target_size[1] // 4))
        deform_feat = self.deform_attn(adapted, x2_ada, x4_ada)
        combined = torch.cat([
            deform_feat,
            F.interpolate(x2_ada, target_size),
            F.interpolate(x4_ada, target_size)
        ], dim=1)

        return self.final_fusion(combined) + self.res_adapter(x1)

if __name__ == "__main__":
    torch.manual_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    in_channels = 3
    height, width = 256, 256
    
    test_input = torch.rand(batch_size, in_channels, height, width, device=device)
    timesteps = torch.randint(0, T, (batch_size,), device=device)
    
    model = DiffusionCondUNet(
        in_channels=in_channels,
        out_channels=1,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        base_ch=32,
        growth_rate=32
    ).to(device)
    
    with torch.no_grad():
        output = model(test_input, timesteps)
    
    test_mask = torch.zeros(1, 1, height, width, device=device)
    center_h, center_w = height // 2, width // 2
    radius = min(height, width) // 4
    for h in range(height):
        for w in range(width):
            if ((h - center_h) ** 2 + (w - center_w) ** 2) < radius ** 2:
                test_mask[0, 0, h, w] = 1.0
    
    t_tensor = torch.tensor([T//2], device=device)
    with torch.no_grad():
        noisy_mask, noise = forward_diffusion_sample(test_mask, t_tensor, device)
