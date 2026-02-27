import math
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from torch.nn import init

step = 2
nC = 28
h = w = 256
W = w + (nC - 1) * step
shift_index = torch.arange(0, step * nC, step).view(-1, 1, 1) + torch.arange(w).view(
    1, 1, -1
)
shift_index = shift_index.expand(nC, 1, w)
shift_index = shift_index.unsqueeze(0).cuda()


def shift(x):
    b, c, h, _ = x.shape
    shift_x = x.new_zeros(b, c, h, W)
    shift_x.scatter_(dim=-1, index=shift_index.expand(b, -1, h, -1), src=x)
    return shift_x


def inv_shift(x):
    b, c, h, _ = x.shape
    inv_shift_x = torch.gather(x, dim=-1, index=shift_index.expand(b, -1, h, -1))
    return inv_shift_x


def R(x, srf=None):
    if srf is not None:
        y = torch.einsum("bcC, bChw->bchw", srf, x)
    else:
        y = x.sum(dim=1)
    return y


def RT(y, srf=None):
    if srf is not None:
        x = torch.einsum("bcC, bchw->bChw", srf, y)
    else:
        x = y.unsqueeze(1).expand(-1, nC, -1, -1)
    return x


def Phi(x, mask):
    return R(shift(x * mask))


def PhiT(y, mask):
    return inv_shift(RT(y)) * mask


def PhiA(E, mask, A):
    # E: bkC
    # mask: b1hw
    # A: bkhw
    x = torch.einsum("bkhw, bkC->bChw", A, E)
    y = Phi(x, mask)
    return y


def PhiAT(y, mask, A):
    # y: bchw
    # mask: b1hw
    # A: bkhw
    x = PhiT(y, mask)
    E = torch.einsum("bChw, bkhw->bkC", x, A)
    return E


def PhiE(A, mask, E):
    # A: bkhw
    # mask: b1hw
    # E: bkC
    x = torch.einsum("bkhw, bkC->bChw", A, E)
    y = Phi(x, mask)
    return y


def PhiET(y, mask, E):
    # y: bchw
    # mask: b1hw
    # E: bkC
    x = PhiT(y, mask)
    A = torch.einsum("bChw, bkC->bkhw", x, E)
    return A


def PhiAMat(mask, A):
    # mask: b1hw
    # A: bkhw
    b, k, h, w = A.shape
    A = A * mask
    A = A.reshape(b * k, h, w)
    A = RT(A)  # b*k c h w
    _, c, h, w = A.shape
    A_shift = shift(A)  # b*k c H W
    _, _, H, W = A_shift.shape
    A_shift = A_shift.reshape(b, k * c, H * W).transpose(-2, -1)  # b, H*W, k*c
    return A_shift


def PhiATMat(mask, A):
    PhiA_mat = PhiAMat(mask, A)
    PhiAT_mat = PhiA_mat.transpose(-2, -1)  # b, k*c, H*W
    return PhiAT_mat


def PhiATPhiAMat(mask, A):
    PhiA_mat = PhiAMat(mask, A)
    PhiAT_mat = PhiA_mat.transpose(-2, -1)  # b, k*c, H*W
    PhiATPhiA_mat = PhiAT_mat @ PhiA_mat  # b, k*c, k*c
    return PhiATPhiA_mat


def InvPhiATPhiAMat(mask, A, lam=None):
    PhiATPhiA_mat = PhiATPhiAMat(mask, A)
    b, l, _ = PhiATPhiA_mat.shape
    if lam is None:
        inv_PhiATPhiA_mat = torch.inverse(PhiATPhiA_mat)
    else:
        I = torch.eye(l).to(PhiATPhiA_mat.device)
        inv_PhiATPhiA_mat = torch.inverse(PhiATPhiA_mat + lam * I)
    return inv_PhiATPhiA_mat


def InvPhiATPhiAMulPhiATy(y, mask, A, E=None, lam=None):
    # y: bchw
    # mask: b1hw
    # A: bkhw
    e = PhiAT(y, mask, A)  # bkC
    if E is not None and lam is not None:
        e = e + lam * E
    b, k, c = e.shape
    PhiATPhiA_inv = InvPhiATPhiAMat(mask, A, lam)  # b, k*C, k*C
    e = e.reshape(b, k * c, 1)
    e = PhiATPhiA_inv @ e  # b, k*C, 1
    e = e.reshape(b, k, c)
    return e
