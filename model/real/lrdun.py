import math
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum
from torch.nn import init


from .physical_op import (
    RT,
    Phi,
    PhiT,
    R,
    inv_shift,
    shift,
    PhiA,
    PhiAT,
    PhiE,
    PhiET,
)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_type="ln"):
        super().__init__()
        self.fn = fn
        self.norm_type = norm_type
        if norm_type == "ln":
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.GroupNorm(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, *args, **kwargs):
        if self.norm_type == "ln":
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class WSSA(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=1, shift=False):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head**-0.5
        self.window_size = window_size
        self.shift = shift

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)
        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def cal_attention(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        h1, h2 = h // self.window_size[0], w // self.window_size[1]
        q, k, v = map(
            lambda t: rearrange(
                t, "b c (h1 h) (h2 w) ->b (h1 h2) c (h w)", h1=h1, h2=h2
            ),
            (q, k, v),
        )
        q *= self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b (h1 h2) c (h w) -> b c (h1 h) (h2 w)", h1=h1, h=h // h1)
        out = self.to_out(out)
        return out

    def forward(self, x):

        w_size = self.window_size
        if self.shift:
            x = x.roll(shifts=w_size[0] // 2, dims=2).roll(
                shifts=w_size[1] // 2, dims=3
            )
        out = self.cal_attention(x)
        if self.shift:
            out = out.roll(shifts=-1 * w_size[1] // 2, dims=3).roll(
                shifts=-1 * w_size[0] // 2, dims=2
            )
        return out


class FFN(nn.Module):
    def __init__(self, dim, mult=2.66):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, bias=False),
        )

    def forward(self, x):

        out = self.net(x)
        return out


class ERB(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=1):
        super().__init__()
        self.WSSA = PreNorm(
            dim,
            WSSA(
                dim=dim,
                window_size=window_size,
                dim_head=dim_head,
                heads=heads,
                shift=False,
            ),
        )
        self.FFN = PreNorm(dim, FFN(dim=dim), norm_type="gn")

    def forward(self, x):

        x = self.WSSA(x) + x
        x = self.FFN(x) + x
        return x


class CMB(nn.Module):
    def __init__(self, dim, k=11):
        super().__init__()

        self.to_a = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Conv2d(dim, dim, k, 1, k//2, groups=dim, bias=False),
        )
        self.to_v = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def cal_attention(self, x):
        a, v = self.to_a(x), self.to_v(x)
        out = self.to_out(a * v)
        return out

    def forward(self, x):
        out = self.cal_attention(x)
        return out


class SAB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.Estimator = nn.Sequential(
            nn.Conv2d(dim, 1, 3, 1, 1, bias=False),
            nn.GELU(),
        )
        self.SW = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight.data, mean=0.0, std=0.02)

    def forward(self, f):
        f = self.conv(f)
        out = self.SW(f) * self.Estimator(f).repeat(1, self.dim, 1, 1)
        out = self.out(out)
        return out


class ARB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.CMB = PreNorm(dim, CMB(dim=dim))
        self.SAB = PreNorm(dim, SAB(dim=dim), norm_type="gn")

    def forward(self, x):

        x = self.CMB(x) + x
        x = self.SAB(x) + x
        return x


class SSRB(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=1):
        super().__init__()

        self.pos = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.SARB = ERB(dim, window_size, dim_head, heads)
        self.SRB = ARB(dim)

    def forward(self, x):

        x = self.pos(x) + x
        x = self.SARB(x)
        x = self.SRB(x)

        return x



class SB(nn.Module):
    def __init__(self, dim, window_size=8, heads=1):
        super().__init__()
        # self.pos = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        self.WSA = PreNorm(
            dim,
            CMB(
                dim=dim, k=11
            ),
        )
        self.FFN_1 = PreNorm(dim, FFN(dim=dim, mult=2.66), norm_type="gn")

        self.SWSA = PreNorm(
            dim,
            CMB(
                dim=dim, k=11
            ),
        )
        self.FFN_2 = PreNorm(dim, FFN(dim=dim, mult=2.66), norm_type="gn")

    def forward(self, x):
        # x = self.pos(x) + x
        x = self.WSA(x) + x
        x = self.FFN_1(x) + x
        x = self.SWSA(x) + x
        x = self.FFN_2(x) + x
        return x


class SSRU(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, rank=11, dim=16):
        super(SSRU, self).__init__()

        # self.mask_embedding = Mask_embedding(dim=in_dim)
        self.embedding = nn.Conv2d(in_dim, dim, 3, 1, 1, bias=False)
        self.down1 = SB(dim=in_dim,  heads=1)
        self.downsample1 = nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False)
        self.down2 = SB(dim=dim * 2,  heads=2)
        self.downsample2 = nn.Conv2d(dim * 2, dim * 4, 4, 2, 1, bias=False)
        self.bottleneck = SB(dim=dim * 4,  heads=4)
        self.upsample2 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, 2)
        self.fusion2 = nn.Conv2d(dim * 4, dim * 2, 1, 1, 0, bias=False)
        self.up2 = SB(dim=dim * 2, heads=2)
        self.upsample1 = nn.ConvTranspose2d(dim * 2, dim, 2, 2)
        self.fusion1 = nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=False)
        self.up1 = SB(dim=dim,  heads=1)
        self.out = nn.Conv2d(dim, out_dim, 3, 1, 1, bias=False)

        self.rank = rank

    def forward(self, x):

        # b, c, h_inp, w_inp = x.shape
        # hb, wb = 16, 16
        # pad_h = (hb - h_inp % hb) % hb
        # pad_w = (wb - w_inp % wb) % wb
        # x_in = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")
        # # mask = F.pad(mask, [0, pad_w, 0, pad_h], mode="reflect")
        x_in = x
        # x = self.mask_embedding(x_in, mask)
        x = self.embedding(x_in)
        # x = x_in
        x1 = self.down1(x)
        x = self.downsample1(x1)
        x2 = self.down2(x)
        x = self.downsample2(x2)
        x = self.bottleneck(x)
        x = self.upsample2(x)
        x = self.fusion2(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        x = self.upsample1(x)
        x = self.fusion1(torch.cat([x, x1], dim=1))
        x = self.up1(x)
        out = self.out(x) + x_in

        out_1 = out[:, : self.rank, :, :]
        out_2 = out[:, self.rank :, :, :]
        return out_1, out_2


class ProxyNetA(nn.Module):
    def __init__(self, in_channels=6, rank=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(
                in_channels * 4, in_channels * 4, 11, 1, 5, groups=in_channels * 4
            ),
            nn.GELU(),
            nn.Conv2d(in_channels * 4, in_channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(in_channels, rank, 1, 1, 0),
        )
        self.ln = nn.LayerNorm(in_channels)
        self.rank = rank
        self.in_channels = in_channels

    def forward(self, x):
        res = x
        # x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = res + self.conv(x)
        # x = F.softmax(x, dim=1)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# class ProxyNetE(nn.Module):
#     def __init__(self, in_channels=28, rank=6):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(rank, in_channels),
#             nn.GELU(),
#             nn.Linear(in_channels, 2 * in_channels),
#             nn.GELU(),
#             nn.Linear(2 * in_channels, in_channels),
#             nn.GELU(),
#             nn.Linear(in_channels, rank),
#         )

#     def forward(self, E):
#         E = E + self.net(E.transpose(-1, -2)).transpose(-1, -2)
#         E = F.normalize(E, p=2, dim=-1)
#         return E


class ProxyNetE(nn.Module):
    def __init__(self, in_channels=28, rank=6, dim=16):
        super().__init__()
        # E 的 shape 是 (b, k, C)
        # nn.Conv1d 需要 (Batch, In_Channels, Length)
        # 我们把 k 视为 "In_Channels"，C 视为 "Length"
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            # 保持 k 不变, 在 C 维度上做 1D 卷积
            nn.Conv1d(
                in_channels=dim,
                out_channels=dim * 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GELU(),
            # 深度卷积 (Depthwise Conv) 来独立处理每个 'k'
            nn.Conv1d(
                in_channels=dim * 2,
                out_channels=dim * 2,
                kernel_size=5,
                padding=2,
                groups=dim * 2,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=dim * 2,
                out_channels=dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )
        self.rank = rank
        self.in_channels = in_channels
        self.dim = dim

    def forward(self, E):
        # E 的 shape 是 (b, k, C)，正好匹配 (N, C_in, L_in)
        E_res = self.net(E)
        E = E + E_res

        E1 = E[:, : self.rank]
        E2 = E[:, self.rank :]

        # E1 = F.normalize(E1, p=2, dim=-1, eps=1e-8)
        QE1, _ = torch.linalg.qr(E1.transpose(-1, -2))
        E1 = QE1.transpose(-1, -2)

        # 归一化 (eps=1e-8 是为了数值稳定性)
        # E = F.normalize(E, p=2, dim=-1, eps=1e-8)
        return E1, E2


class Para_Estimator(nn.Module):
    def __init__(self, in_nc=28, out_nc=2, channel=32):
        super(Para_Estimator, self).__init__()
        self.fusion = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.bias = nn.Parameter(torch.FloatTensor([1.0]))
        self.avpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, out_nc, 1, padding=0, bias=False),
        )
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x):

        x = self.relu(self.fusion(x))
        x = self.avpool(x)
        x = self.mlp(x) + self.bias
        return x


class InitE(nn.Module):
    def __init__(self, in_channels=28, rank=6):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(in_channels, rank * in_channels, 1, 1, 0),
        )
        self.rank = rank
        self.in_channels = in_channels

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.view(x.size(0), self.rank, self.in_channels)
        # x = F.normalize(x, p=2, dim=-1)
        Q, R = torch.linalg.qr(x.transpose(-1, -2))  # QR 分解
        x = Q.transpose(-1, -2)  # 返回正交矩阵 Q
        return x


class InitA(nn.Module):
    def __init__(self, in_channels=28, rank=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, rank, 1, 1, 0),
        )

    def forward(self, x):
        x = self.net(x)
        # x = F.softmax(x, dim=1)
        return x


class LRDUN(torch.nn.Module):
    def __init__(self, stage=3, bands=28, rank=6, dim=16):
        super().__init__()
        self.stage = stage
        self.nC = bands
        self.rank = rank
        self.size = 660
        self.init_x = nn.Conv2d(self.nC + 1, self.nC, 1, 1, 0)
        self.init_E = InitE(in_channels=bands, rank=dim)
        self.init_A = InitA(in_channels=bands, rank=dim)
        self.proxy_net_Es = nn.ModuleList()
        self.proxy_net_As = nn.ModuleList()
        self.para_nets = nn.ModuleList()
        self.eta_As = nn.Parameter(torch.ones(stage) * 1e-4)
        self.eta_Es = nn.Parameter(torch.ones(stage) * 1e-4)
        for i in range(stage):
            # self.proxy_net_As.append(ProxyNetA(in_channels=rank, rank=rank))
            self.proxy_net_As.append(SSRU(in_dim=dim, out_dim=dim, rank=rank, dim=dim))
            self.proxy_net_Es.append(ProxyNetE(in_channels=bands, rank=rank, dim=dim))
            self.para_nets.append(Para_Estimator(in_nc=bands, out_nc=2, channel=32))

    def forward(self, inputs):
        y = inputs["CASSI_measure"]
        Phi_s = inputs["Phi_s"]
        mask = inputs["CASSI_mask"].unsqueeze(1)
        # Phi, PhiPhiT = input_mask
        x = inv_shift(RT(y)) / self.nC * 2
        x = self.init_x(torch.cat([x, mask], dim=1))
        E = self.init_E(x)
        A = self.init_A(x)
        E, EV = E[:, : self.rank], E[:, self.rank :]
        A, AV = A[:, : self.rank], A[:, self.rank :]
        out = []
        for i in range(self.stage):
            para_net = self.para_nets[i]
            proxy_net_E = self.proxy_net_Es[i]
            proxy_net_A = self.proxy_net_As[i]
            # eta_E, eta_A = para_net(x).chunk(2, dim=1)  # b2hw->b1hw,b1hw
            # eta_E = eta_E.squeeze(1)
            # eta_A = eta_E = 1e-3
            eta_E = self.eta_Es[i]
            eta_A = self.eta_As[i]
            E = E - eta_E * PhiAT(PhiA(E, mask, A) - y, mask, A)
            E, EV = proxy_net_E(torch.cat([E, EV], dim=1))
            A = A - eta_A * PhiET(PhiE(A, mask, E) - y, mask, E)
            A, AV = proxy_net_A(torch.cat([A, AV], dim=1))
            x = torch.einsum("bkC, bkhw->bChw", E, A)
            out.append(x)
        outputs = {
            "pred": out[-1],
            "immediate_results": out,
        }
        return outputs
