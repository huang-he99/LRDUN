import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from math import exp
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from model.simu.lrdun import LRDUN
from model.simu.physical_op import RT, Phi, PhiT, R
from safetensors.torch import load_file
from torch.autograd import Variable
from tqdm import tqdm


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


# We find that this calculation method is more close to DGSMP's.
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))


def sam(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """
    num = 0
    sum_sam = 0
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            if np.linalg.norm(tmp_true) != 0 and np.linalg.norm(tmp_pred) != 0:
                sum_sam += np.arccos(
                    np.inner(tmp_pred, tmp_true) / (np.linalg.norm(tmp_true) * np.linalg.norm(tmp_pred)))
                num += 1
    sam_deg = (sum_sam / num) * 180 / np.pi
    # 
    return sam_deg

stg_num = 9

checkpoint_path = Path(f'./checkpoint/simu/lrdun_{stg_num}stg.safetensors')

hsi_dir_path = Path('./data/simu')
hsi_path_list = sorted(list(hsi_dir_path.glob('*.mat')))

mask_data = sio.loadmat('./data/mask/simu/mask.mat')
mask_np = mask_data['mask']
mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).cuda()

Phi_s = Phi(RT(mask_tensor), mask_tensor)


model = LRDUN(stage=stg_num, dim=16, rank=11, bands=28).cuda()
checkpoint = load_file(checkpoint_path)
model.load_state_dict(checkpoint)

save_dir_path = Path(f'./results/simu/lrdun_{stg_num}stg/')
save_dir_path.mkdir(exist_ok=True, parents=True)

psnr_list, ssim_list, sam_list = [], [], []

for hsi_path in tqdm(hsi_path_list):
    hsi_data = sio.loadmat(hsi_path)  # [H, W, C]
    hsi_np = hsi_data['img']  # [H, W, C]
    hsi_tensor = torch.from_numpy(hsi_np).float().unsqueeze(0).permute(0, 3, 1, 2).cuda()  # [1, C, H, W]
    meas_tensor = Phi(hsi_tensor, mask_tensor)  # [1, H, W]
    input = dict(CASSI_measure=meas_tensor, Phi_s=Phi_s, CASSI_mask=mask_tensor)
    with torch.no_grad():
        output = model(input)
    pred = output['pred']

    pred_tensor = pred.squeeze(0) # [C, H, W]
    gt_tensor = hsi_tensor.squeeze(0) # [C, H, W]

    psnr_val = torch_psnr(pred_tensor, gt_tensor).item()
    ssim_val = torch_ssim(pred_tensor, gt_tensor).item()
    sam_val = sam(gt_tensor.cpu().numpy(), pred_tensor.cpu().numpy())

    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)
    sam_list.append(sam_val)

    print(f'Image: {hsi_path.stem}, PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.3f}, SAM: {sam_val:.2f}')

    save_path = save_dir_path / f'{hsi_path.stem}.npy'
    np.save(save_path, pred_tensor.cpu().numpy().transpose(1, 2, 0))  # [H, W, C]

psnr_mean = np.mean(np.asarray(psnr_list))
ssim_mean = np.mean(np.asarray(ssim_list))
sam_mean = np.mean(np.asarray(sam_list))

print(f'Mean PSNR: {psnr_mean:.2f}, Mean SSIM: {ssim_mean:.3f}, Mean SAM: {sam_mean:.2f}')
