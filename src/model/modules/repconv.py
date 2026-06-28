import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_kxk(w: torch.Tensor, target_k: int) -> torch.Tensor:
    _, _, h, wk = w.shape
    ph, pw = target_k - h, target_k - wk
    pt, pb = ph // 2, ph - ph // 2
    pl, pr = pw // 2, pw - pw // 2
    return F.pad(w, [pl, pr, pt, pb])

def fold_1x1_into_kxk(w1: torch.Tensor, w2: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return F.conv2d(w2, w1.permute(1, 0, 2, 3))
    
    icpg = w1.shape[0] // groups
    ocpg = w2.shape[0] // groups
    w1_T = w1.permute(1, 0, 2, 3)
    slices = []
    for g in range(groups):
        slices.append(F.conv2d(
            w2[g * ocpg:(g + 1) * ocpg],
            w1_T[:, g * icpg:(g + 1) * icpg]
        ))
    return torch.cat(slices, dim=0)

def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    w = conv.weight
    device = w.device
    dtype = w.dtype

    b = conv.bias if conv.bias is not None else torch.zeros(
        w.size(0), device=device, dtype=dtype
    )

    running_mean = bn.running_mean.to(device)
    running_var  = bn.running_var.to(device)
    gamma        = bn.weight.to(device)
    beta         = bn.bias.to(device)

    eps = bn.eps
    std = torch.sqrt(running_var + eps)

    scale = gamma / std
    scale = scale.reshape(-1, 1, 1, 1)

    w_fused = w * scale

    b_fused = (b - running_mean) / std * gamma + beta

    return w_fused, b_fused

class ConvBN(nn.Module):
    """Conv2d + BatchNorm2d"""
    def __init__(self, in_c, out_c, kernel_size, padding, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))

    def get_fused_weight_bias(self):
        return fuse_conv_bn(self.conv, self.bn)

# =============================================================================
class RepConv3BN(nn.Module):
    """RepConv 3x3: 3x3, 3x1, 1x3, 1x1, 1x1->3x3"""
    def __init__(self, in_channels, out_channels, groups=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        
        self.reparam = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=groups, bias=True)
        
        if not deploy:
            self.br_3x3 = ConvBN(in_channels, out_channels, 3, 1, groups)
            self.br_3x1 = ConvBN(in_channels, out_channels, (3, 1), (1, 0), groups)
            self.br_1x3 = ConvBN(in_channels, out_channels, (1, 3), (0, 1), groups)
            self.br_1x1 = ConvBN(in_channels, out_channels, 1, 0, groups)
            
            # 1x1 -> 3x3
            self.br_seq_1x1 = nn.Conv2d(in_channels, in_channels, 1, groups=groups, bias=False)
            self.br_seq_3x3 = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=groups, bias=False)
            self.br_seq_bn = nn.BatchNorm2d(out_channels)

    def fuse(self, delete_branches: bool = True):
        if self.deploy: return
        device = self.reparam.weight.device
        W, B = 0, 0
        
        # Fuse
        for branch in [self.br_3x3, self.br_3x1, self.br_1x3, self.br_1x1]:
            w, b = branch.get_fused_weight_bias()
            W += pad_to_kxk(w, 3).to(device)
            B += b.to(device)
            
        w_seq = fold_1x1_into_kxk(self.br_seq_1x1.weight.detach(), self.br_seq_3x3.weight.detach(), self.groups).to(device)
        
        std = torch.sqrt(self.br_seq_bn.running_var.to(device) + self.br_seq_bn.eps)
        scale = self.br_seq_bn.weight.to(device) / std
        
        W += w_seq * scale.reshape(-1, 1, 1, 1)
        B += -self.br_seq_bn.running_mean.to(device) / std * self.br_seq_bn.weight.to(device) + self.br_seq_bn.bias.to(device)

        self.reparam.weight.data.copy_(W)
        self.reparam.bias.data.copy_(B)
        
        if delete_branches:
            for name in ['br_3x3', 'br_3x1', 'br_1x3', 'br_1x1', 'br_seq_1x1', 'br_seq_3x3', 'br_seq_bn']:
                delattr(self, name)
        self.deploy = True

    def forward(self, x):
        if self.deploy: return self.reparam(x)
        seq_out = self.br_seq_bn(self.br_seq_3x3(self.br_seq_1x1(x)))
        return self.br_3x3(x) + self.br_3x1(x) + self.br_1x3(x) + self.br_1x1(x) + seq_out


class RepConv7BN(nn.Module):
    """RepConv 7x7: 7×7 | 7×1 | 1×7 | 7×5 | 5×7 | 5×5 | 1×5 | 5×1 | Sequential(1×1→7×7)"""
    def __init__(self, in_channels, out_channels, groups=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        
        self.reparam = nn.Conv2d(in_channels, out_channels, 7, padding=3, groups=groups, bias=True)
        
        if not deploy:
            self.br_7x7 = ConvBN(in_channels, out_channels, 7, 3, groups)
            self.br_7x1 = ConvBN(in_channels, out_channels, (7, 1), (3, 0), groups)
            self.br_1x7 = ConvBN(in_channels, out_channels, (1, 7), (0, 3), groups)
            self.br_7x5 = ConvBN(in_channels, out_channels, (7, 5), (3, 2), groups)
            self.br_5x7 = ConvBN(in_channels, out_channels, (5, 7), (2, 3), groups)
            self.br_5x5 = ConvBN(in_channels, out_channels, 5, 2, groups)
            self.br_1x5 = ConvBN(in_channels, out_channels, (1, 5), (0, 2), groups)
            self.br_5x1 = ConvBN(in_channels, out_channels, (5, 1), (2, 0), groups)
            
            self.br_seq_1x1 = nn.Conv2d(in_channels, in_channels, 1, groups=groups, bias=False)
            self.br_seq_7x7 = nn.Conv2d(in_channels, out_channels, 7, padding=3, groups=groups, bias=False)
            self.br_seq_bn = nn.BatchNorm2d(out_channels)

    def fuse(self, delete_branches: bool = True):
        if self.deploy: return
        device = self.reparam.weight.device
        W, B = 0, 0
        
        branches = [self.br_7x7, self.br_7x1, self.br_1x7, self.br_7x5, 
                    self.br_5x7, self.br_5x5, self.br_1x5, self.br_5x1]
        
        for branch in branches:
            w, b = branch.get_fused_weight_bias()
            W += pad_to_kxk(w, 7).to(device)
            B += b
            
        w_seq = fold_1x1_into_kxk(self.br_seq_1x1.weight.detach(), self.br_seq_7x7.weight.detach(), self.groups).to(device)
        std = torch.sqrt(self.br_seq_bn.running_var.to(device) + self.br_seq_bn.eps)
        scale = self.br_seq_bn.weight.to(device) / std
        W += w_seq * scale.reshape(-1, 1, 1, 1)
        B += -self.br_seq_bn.running_mean.to(device) / std * self.br_seq_bn.weight.to(device) + self.br_seq_bn.bias.to(device)

        self.reparam.weight.data.copy_(W)
        self.reparam.bias.data.copy_(B)
        
        if delete_branches:
            for name in ['br_7x7', 'br_7x1', 'br_1x7', 'br_7x5', 'br_5x7', 'br_5x5', 'br_1x5', 'br_5x1', 'br_seq_1x1', 'br_seq_7x7', 'br_seq_bn']:
                delattr(self, name)
        self.deploy = True

    def forward(self, x):
        if self.deploy: return self.reparam(x)
        seq_out = self.br_seq_bn(self.br_seq_7x7(self.br_seq_1x1(x)))
        return (self.br_7x7(x) + self.br_7x1(x) + self.br_1x7(x) + 
                self.br_7x5(x) + self.br_5x7(x) + self.br_5x5(x) + 
                self.br_1x5(x) + self.br_5x1(x) + seq_out)

# =============================================================================
# DEPTHWISE & POINTWISE MODULES (INCLUDED BATCHNORM)

class DWRepConv3BN(RepConv3BN):
    """Depthwise RepConv 3x3. Inherit RepConv3BN by setting groups=channels and optimize weight multiplication."""
    def __init__(self, channels, deploy=False):
        super().__init__(in_channels=channels, out_channels=channels, groups=channels, deploy=deploy)
    
    def fuse(self, delete_branches: bool = True):
        if self.deploy: return
        device = self.reparam.weight.device
        W, B = 0, 0
        
        for branch in [self.br_3x3, self.br_3x1, self.br_1x3, self.br_1x1]:
            w, b = branch.get_fused_weight_bias()
            W += pad_to_kxk(w, 3)
            B += b
            
        w1 = self.br_seq_1x1.weight.detach() # [C, 1, 1, 1]
        w2 = self.br_seq_3x3.weight.detach() # [C, 1, 3, 3]
        w_seq = (w1 * w2).to(device)
        
        std = torch.sqrt(self.br_seq_bn.running_var + self.br_seq_bn.eps)
        scale = self.br_seq_bn.weight / std
        W += w_seq * scale.reshape(-1, 1, 1, 1)
        B += -self.br_seq_bn.running_mean / std * self.br_seq_bn.weight + self.br_seq_bn.bias

        self.reparam.weight.data.copy_(W)
        self.reparam.bias.data.copy_(B)
        
        if delete_branches:
            for name in ['br_3x3', 'br_3x1', 'br_1x3', 'br_1x1', 'br_seq_1x1', 'br_seq_3x3', 'br_seq_bn']:
                delattr(self, name)
        self.deploy = True


class DWRepConv7BN(RepConv7BN):
    """Depthwise RepConv 3x3. Inherit RepConv7BN by setting groups=channels and optimize weight multiplication."""
    def __init__(self, channels, deploy=False):
        super().__init__(in_channels=channels, out_channels=channels, groups=channels, deploy=deploy)
    
    def fuse(self, delete_branches: bool = True):
        if self.deploy: return
        W, B = 0, 0
        
        branches = [self.br_7x7, self.br_7x1, self.br_1x7, self.br_7x5, 
                    self.br_5x7, self.br_5x5, self.br_1x5, self.br_5x1]
        
        for branch in branches:
            w, b = branch.get_fused_weight_bias()
            W += pad_to_kxk(w, 7)
            B += b
            
        w1 = self.br_seq_1x1.weight.detach() 
        w2 = self.br_seq_7x7.weight.detach() 
        w_seq = w1 * w2
        
        std = torch.sqrt(self.br_seq_bn.running_var + self.br_seq_bn.eps)
        scale = self.br_seq_bn.weight / std
        W += w_seq * scale.reshape(-1, 1, 1, 1)
        B += -self.br_seq_bn.running_mean / std * self.br_seq_bn.weight + self.br_seq_bn.bias

        self.reparam.weight.data.copy_(W)
        self.reparam.bias.data.copy_(B)
        
        if delete_branches:
            for name in ['br_7x7', 'br_7x1', 'br_1x7', 'br_7x5', 'br_5x7', 'br_5x5', 'br_1x5', 'br_5x1', 'br_seq_1x1', 'br_seq_7x7', 'br_seq_bn']:
                delattr(self, name)
        self.deploy = True


class RepPointwiseBN(nn.Module):
    """1x1 Convolution + Identity (nếu in_channels == out_channels) BatchNorm"""
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_identity = (in_channels == out_channels)

        self.reparam = nn.Conv2d(in_channels, out_channels, 1, bias=True)

        if not deploy:
            self.br_1x1 = ConvBN(in_channels, out_channels, 1, 0)
            if self.has_identity:
                self.br_id_bn = nn.BatchNorm2d(in_channels)

    def fuse(self, delete_branches: bool = True):
        if self.deploy: return

        W, B = self.br_1x1.get_fused_weight_bias()

        if self.has_identity:
            W_id = torch.zeros((self.in_channels, self.in_channels, 1, 1), device=W.device)
            for i in range(self.in_channels):
                W_id[i, i, 0, 0] = 1.0
            
            std = torch.sqrt(self.br_id_bn.running_var + self.br_id_bn.eps)
            scale = self.br_id_bn.weight / std
            W_id_fused = W_id * scale.reshape(-1, 1, 1, 1)
            B_id_fused = -self.br_id_bn.running_mean / std * self.br_id_bn.weight + self.br_id_bn.bias
            
            W += W_id_fused
            B += B_id_fused

        self.reparam.weight.data.copy_(W)
        self.reparam.bias.data.copy_(B)

        if delete_branches:
            delattr(self, 'br_1x1')
            if self.has_identity: delattr(self, 'br_id_bn')
            
        self.deploy = True

    def forward(self, x):
        if self.deploy: return self.reparam(x)
        out = self.br_1x1(x)
        if self.has_identity:
            out += self.br_id_bn(x)
        return out

# SEPARABLE CONVOLUTION MODULES
class RepDWSeparable3BN(nn.Module):
    def __init__(self, in_channels, out_channels, expand=2, deploy=False):
        super().__init__()

        self.use_skip = (in_channels == out_channels)
        hidden = in_channels * expand

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU()
        )

        self.dw = DWRepConv3BN(hidden, deploy=deploy)

        self.project = nn.Sequential(nn.Conv2d(hidden, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))

    def fuse(self, delete_branches: bool = True):
        self.dw.fuse(delete_branches)

    def forward(self, x):
        identity = x

        x = self.expand(x)
        x = self.dw(x)
        x = self.project(x)

        if self.use_skip:
            x = x + identity

        return x

class RepDWSeparable7BN(nn.Module):
    def __init__(self, in_channels, out_channels, expand=4, deploy=False):
        super().__init__()

        self.use_skip = (in_channels == out_channels)
        hidden = in_channels * expand

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU()
        )

        self.dw = DWRepConv7BN(hidden, deploy=deploy)

        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_channels, 1, bias=False), 
            nn.BatchNorm2d(out_channels)
        )

    def fuse(self, delete_branches: bool = True):
        self.dw.fuse(delete_branches)

    def forward(self, x):
        identity = x

        x = self.expand(x)
        x = self.dw(x)
        x = self.project(x)

        if self.use_skip:
            x = x + identity

        return x