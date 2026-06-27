import torch
import torch.nn as nn
import torch.nn.functional as F


class RepConv3BN(nn.Module):
    """
    Branches: 3×3 | 3×1 | 1×3 | 1×1 | Sequential(1×1→3×3)
    """

    def __init__(self, in_channels, out_channels, groups=1, deploy=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.groups       = groups
        self.deploy       = deploy

        self.reparam = nn.Conv2d(
            in_channels, out_channels, 3,
            padding=1, groups=groups, bias=True
        )

        if not deploy:
            def make_conv(k, p):
                return nn.Conv2d(in_channels, out_channels, k,
                                 padding=p, groups=groups, bias=True)

            self.branches = nn.ModuleList([
                make_conv(3,       1     ),   # 0: 3×3
                make_conv((3, 1),  (1, 0)),   # 1: 3×1
                make_conv((1, 3),  (0, 1)),   # 2: 1×3
                make_conv(1,       0     ),   # 3: 1×1
                nn.Sequential(               # 4: 1×1 → 3×3
                    nn.Conv2d(in_channels, in_channels,  1,
                              groups=groups, bias=False),
                    nn.Conv2d(in_channels, out_channels, 3,
                              padding=1, groups=groups, bias=False),
                )
            ])

            self.bns = nn.ModuleList([
                nn.BatchNorm2d(out_channels) for _ in range(5)
            ])

            self.branch_weight = nn.Parameter(torch.ones(5))

    # ── helpers ──────────────────────────────────────────────

    def _branch_forward(self, x, i):
        return self.bns[i](self.branches[i](x))

    def _delete_branches(self):
        for name in ('branches', 'bns', 'branch_weight'):
            if hasattr(self, name):
                delattr(self, name)

    def _pad_to_3x3(self, w: torch.Tensor) -> torch.Tensor:
        _, _, h, wk = w.shape
        ph, pw  = 3 - h, 3 - wk
        pt, pb  = ph // 2, ph - ph // 2
        pl, pr  = pw // 2, pw - pw // 2
        return F.pad(w, [pl, pr, pt, pb])

    @staticmethod
    def _fuse_conv_bn(conv_w, conv_b, bn):
        """Fold BN vào conv: w' = w*γ/std,  b' = (b-μ)/std*γ + β"""
        b = conv_b if conv_b is not None else torch.zeros(
            conv_w.size(0), device=conv_w.device, dtype=conv_w.dtype)
        std = torch.sqrt(bn.running_var + bn.eps)
        scale = bn.weight / std
        w_f = conv_w * scale.reshape(-1, 1, 1, 1)
        b_f = (b - bn.running_mean) / std * bn.weight + bn.bias
        return w_f, b_f

    @staticmethod
    def _fold_1x1_into_kxk(w1: torch.Tensor, w2: torch.Tensor,
                            groups: int = 1) -> torch.Tensor:
        """
        Fold sequential Conv1×1(w1) → ConvK×K(w2) thành một ConvK×K.
        w1: [C_in, C_in/g, 1, 1]   (1×1 conv, bias=False)
        w2: [C_out, C_in/g, K, K]  (K×K conv, bias=False)
        Kết quả: [C_out, C_in/g, K, K]

        Toán học (groups=1):
          y = w2 * (w1 * x) = (w2 ⊛ w1) * x
          w_seq[o,i,h,w] = Σ_k w2[o,k,h,w] * w1[k,i,0,0]
        Dùng F.conv2d(w2, w1.permute(1,0,2,3)) để vectorise phép này.
        """
        if groups == 1:
            return F.conv2d(w2, w1.permute(1, 0, 2, 3))
        # grouped: xử lý từng group độc lập
        icpg = w1.shape[0] // groups    # C_in / g
        ocpg = w2.shape[0] // groups    # C_out / g
        slices = []
        w1_T = w1.permute(1, 0, 2, 3)  # [C_in/g, C_in, 1, 1]
        for g in range(groups):
            w1_g = w1_T[:, g * icpg:(g + 1) * icpg]   # [C_in/g, icpg, 1,1]
            w2_g = w2[g * ocpg:(g + 1) * ocpg]         # [ocpg, C_in/g, K,K]
            slices.append(F.conv2d(w2_g, w1_g))
        return torch.cat(slices, dim=0)

    # ── fuse ─────────────────────────────────────────────────

    def fuse(self, delete_branches: bool = True):
        if self.deploy:
            return

        self.eval()

        W = torch.zeros_like(self.reparam.weight)
        B = torch.zeros_like(self.reparam.bias)

        w_scale = torch.softmax(self.branch_weight.detach(), dim=0)

        # Branch 0-3: conv+BN
        for i in range(4):
            conv = self.branches[i]
            w, b = self._fuse_conv_bn(conv.weight, conv.bias, self.bns[i])
            w    = self._pad_to_3x3(w)
            W   += w_scale[i] * w
            B   += w_scale[i] * b

        # Branch 4: sequential 1×1 → 3×3
        w1 = self.branches[4][0].weight   # [C_in,  C_in/g, 1, 1]
        w2 = self.branches[4][1].weight   # [C_out, C_in/g, 3, 3]
        w_seq = self._fold_1x1_into_kxk(
            w1.detach(), w2.detach(), self.groups)
        w_seq, b_seq = self._fuse_conv_bn(w_seq, None, self.bns[4])
        W += w_scale[4] * w_seq
        B += w_scale[4] * b_seq

        self.reparam.weight.data.copy_(W)
        self.reparam.bias.data.copy_(B)

        if delete_branches:
            self._delete_branches()

        self.deploy = True

    # ── forward ──────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.reparam(x)

        w = torch.softmax(self.branch_weight, dim=0)
        out = sum(w[i] * self._branch_forward(x, i) for i in range(5))
        return out


class RepConv3(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, deploy=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.groups       = groups
        self.deploy       = deploy

        self.reparam = nn.Conv2d(
            in_channels, out_channels, 3,
            padding=1, groups=groups, bias=True
        )

        if not deploy:
            def _c(k, p):
                return nn.Conv2d(in_channels, out_channels, k,
                                 padding=p, groups=groups, bias=True)

            self.branches = nn.ModuleList([
                _c(3,      1     ),    # 0: 3×3
                _c((3, 1), (1, 0)),    # 1: 3×1
                _c((1, 3), (0, 1)),    # 2: 1×3
                _c(1,      0     ),    # 3: 1×1
                nn.Sequential(         # 4: 1×1 → 3×3
                    nn.Conv2d(in_channels, in_channels,  1,
                              groups=groups, bias=False),
                    nn.Conv2d(in_channels, out_channels, 3,
                              padding=1, groups=groups, bias=True),
                ),
            ])

    # ── helpers ──────────────────────────────────────────────

    def _delete_branches(self):
        if hasattr(self, 'branches'):
            del self.branches

    def _pad_to_3x3(self, w: torch.Tensor) -> torch.Tensor:
        _, _, h, wk = w.shape
        ph, pw = 3 - h, 3 - wk
        pt, pb = ph // 2, ph - ph // 2
        pl, pr = pw // 2, pw - pw // 2
        return F.pad(w, [pl, pr, pt, pb])

    @staticmethod
    def _fold_1x1_into_3x3(w1: torch.Tensor, w2: torch.Tensor,
                            groups: int = 1) -> torch.Tensor:
        if groups == 1:
            return F.conv2d(w2, w1.permute(1, 0, 2, 3))
        icpg = w1.shape[0] // groups
        ocpg = w2.shape[0] // groups
        w1_T = w1.permute(1, 0, 2, 3)
        slices = []
        for g in range(groups):
            slices.append(F.conv2d(
                w2[g * ocpg:(g + 1) * ocpg],
                w1_T[:, g * icpg:(g + 1) * icpg],
            ))
        return torch.cat(slices, dim=0)


    def fuse(self, delete_branches: bool = True):
        if self.deploy:
            return

        W = torch.zeros_like(self.reparam.weight)
        B = torch.zeros_like(self.reparam.bias)

        for i in range(4):
            conv = self.branches[i]
            W += self._pad_to_3x3(conv.weight.detach())
            B += conv.bias.detach()

        w1   = self.branches[4][0].weight.detach()   # [C_in,  C_in/g, 1, 1]
        w2   = self.branches[4][1].weight.detach()   # [C_out, C_in/g, 3, 3]
        b2   = self.branches[4][1].bias.detach()     # [C_out]
        W   += self._fold_1x1_into_3x3(w1, w2, self.groups)
        B   += b2

        self.reparam.weight.data.copy_(W)
        self.reparam.bias.data.copy_(B)

        if delete_branches:
            self._delete_branches()

        self.deploy = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.reparam(x)

        out = (
            self.branches[0](x)              # 3×3
            + self.branches[1](x)            # 3×1
            + self.branches[2](x)            # 1×3
            + self.branches[3](x)            # 1×1
            + self.branches[4](x)            # 1×1 → 3×3
        )
        return out


class RepConv7(nn.Module):
    """
    Branches: 7×7 | 7×1 | 1×7 | 7×5 | 5×7 | 5×5 | 1×5 | 5×1
              | Sequential(1×1→7×7)
    """

    def __init__(self, in_channels, out_channels, groups=1, deploy=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.groups       = groups
        self.deploy       = deploy

        # [FIX-5] bias=True tường minh
        self.reparam = nn.Conv2d(
            in_channels, out_channels, 7,
            padding=3, groups=groups, bias=True
        )

        if not deploy:
            def _c(k, p):
                return nn.Conv2d(in_channels, out_channels, k,
                                 padding=p, groups=groups, bias=True)

            self.conv_7x7 = _c(7,       3      )
            self.conv_7x1 = _c((7, 1),  (3, 0) )
            self.conv_1x7 = _c((1, 7),  (0, 3) )
            self.conv_7x5 = _c((7, 5),  (3, 2) )
            self.conv_5x7 = _c((5, 7),  (2, 3) )
            self.conv_5x5 = _c(5,       2      )
            self.conv_1x5 = _c((1, 5),  (0, 2) )
            self.conv_5x1 = _c((5, 1),  (2, 0) )

            self.conv_1x1_branch = nn.Conv2d(
                in_channels, in_channels, 1,
                groups=groups, bias=False)
            self.conv_7x7_branch = nn.Conv2d(
                in_channels, out_channels, 7,
                padding=3, groups=groups, bias=False)


    def _delete_branches(self):
        for name in (
            'conv_7x7', 'conv_7x1', 'conv_1x7',
            'conv_7x5', 'conv_5x7',
            'conv_5x5',
            'conv_1x5', 'conv_5x1',
            'conv_1x1_branch', 'conv_7x7_branch',
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _pad_to_7x7(self, w: torch.Tensor) -> torch.Tensor:
        _, _, h, wk = w.shape
        ph, pw  = 7 - h, 7 - wk
        pt, pb  = ph // 2, ph - ph // 2
        pl, pr  = pw // 2, pw - pw // 2
        return F.pad(w, [pl, pr, pt, pb])

    @staticmethod
    def _fold_1x1_into_kxk(w1: torch.Tensor, w2: torch.Tensor,
                            groups: int = 1) -> torch.Tensor:
        """Giống RepConv3._fold_1x1_into_kxk — xem docstring đó."""
        if groups == 1:
            return F.conv2d(w2, w1.permute(1, 0, 2, 3))
        icpg = w1.shape[0] // groups
        ocpg = w2.shape[0] // groups
        w1_T = w1.permute(1, 0, 2, 3)
        slices = []
        for g in range(groups):
            slices.append(F.conv2d(
                w2[g * ocpg:(g + 1) * ocpg],
                w1_T[:, g * icpg:(g + 1) * icpg],
            ))
        return torch.cat(slices, dim=0)


    def fuse(self, delete_branches: bool = True):
        if self.deploy:
            return

        self.eval()

        W = torch.zeros_like(self.reparam.weight)
        B = torch.zeros_like(self.reparam.bias)

        def _get(conv):
            b = conv.bias if conv.bias is not None else torch.zeros(
                self.out_channels, device=conv.weight.device)
            return conv.weight.detach(), b.detach()

        for conv in (
            self.conv_7x7,
            self.conv_7x1, self.conv_1x7,
            self.conv_7x5, self.conv_5x7,
            self.conv_5x5,
            self.conv_1x5, self.conv_5x1,
        ):
            w, b = _get(conv)
            W += self._pad_to_7x7(w)
            B += b

        w_seq = self._fold_1x1_into_kxk(
            self.conv_1x1_branch.weight.detach(),
            self.conv_7x7_branch.weight.detach(),
            self.groups,
        )
        W += w_seq

        self.reparam.weight.data.copy_(W)
        self.reparam.bias.data.copy_(B)

        if delete_branches:
            self._delete_branches()

        self.deploy = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.reparam(x)

        return (
            self.conv_7x7(x)
            + self.conv_7x1(x) + self.conv_1x7(x)
            + self.conv_7x5(x) + self.conv_5x7(x)
            + self.conv_5x5(x)
            + self.conv_1x5(x) + self.conv_5x1(x)
            + self.conv_7x7_branch(self.conv_1x1_branch(x))
        )


class DWRepConv3(nn.Module):
    def __init__(self, channels, deploy=False):
        super().__init__()
        self.channels = channels
        self.deploy   = deploy

        self.reparam = nn.Conv2d(
            channels, channels, 3,
            padding=1, groups=channels, bias=True
        )

        if not deploy:
            def _c(k, p):
                return nn.Conv2d(channels, channels, k,
                                 padding=p, groups=channels, bias=True)

            self.branches = nn.ModuleList([
                _c(3,      1     ),    # 0: 3×3
                _c((3, 1), (1, 0)),    # 1: 3×1
                _c((1, 3), (0, 1)),    # 2: 1×3
                _c(1,      0     ),    # 3: 1×1
                nn.Sequential(         # 4: 1×1 → 3×3
                    nn.Conv2d(channels, channels, 1, groups=channels, bias=False),
                    nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True),
                ),
            ])

    def _pad_to_3x3(self, w: torch.Tensor) -> torch.Tensor:
        _, _, h, wk = w.shape
        ph, pw = 3 - h, 3 - wk
        pt, pb = ph // 2, ph - ph // 2
        pl, pr = pw // 2, pw - pw // 2
        return F.pad(w, [pl, pr, pt, pb])

    def fuse(self, delete_branches: bool = True):
        if self.deploy:
            return

        W = torch.zeros_like(self.reparam.weight)
        B = torch.zeros_like(self.reparam.bias)

        # Gộp nhánh 0-3 (Cộng trực tiếp trọng số và bias)
        for i in range(4):
            conv = self.branches[i]
            W += self._pad_to_3x3(conv.weight.detach())
            B += conv.bias.detach()

        # Gộp nhánh 4 (Sequential) - Dùng phép nhân ma trận tối ưu cho DW
        w1 = self.branches[4][0].weight.detach()
        w2 = self.branches[4][1].weight.detach()
        b2 = self.branches[4][1].bias.detach()
        
        W += w1 * w2
        B += b2

        self.reparam.weight.data.copy_(W)
        self.reparam.bias.data.copy_(B)

        if delete_branches:
            delattr(self, 'branches')
            
        self.deploy = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.reparam(x)
        return sum(branch(x) for branch in self.branches)


# 2. RepPointwise (Pointwise 1x1 - Không BN)
class RepPointwise(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy

        self.reparam = nn.Conv2d(in_channels, out_channels, 1, bias=True)

        if not deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=True)
            self.has_identity = (in_channels == out_channels)

    def fuse(self, delete_branches: bool = True):
        if self.deploy: return

        W = self.conv.weight.detach().clone()
        B = self.conv.bias.detach().clone()

        if self.has_identity:
            W_id = torch.zeros_like(W)
            for i in range(self.in_channels):
                W_id[i, i, 0, 0] = 1.0
            
            W += W_id

        self.reparam.weight.data.copy_(W)
        self.reparam.bias.data.copy_(B)

        if delete_branches:
            delattr(self, 'conv')
            
        self.deploy = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.reparam(x)
        
        out = self.conv(x)
        if self.has_identity:
            out += x
        return out


class RepDWSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        
        # Depthwise -> ReLU -> Pointwise
        self.dw = DWRepConv3(channels=in_channels, deploy=deploy)
        self.act = nn.ReLU(inplace=True)
        self.pw = RepPointwise(in_channels, out_channels, deploy=deploy)

    def fuse(self, delete_branches: bool = True):
        self.dw.fuse(delete_branches)
        self.pw.fuse(delete_branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.act(x)
        x = self.pw(x)
        return x