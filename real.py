import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from model.real.lrdun import LRDUN
from model.real.physical_op import RT, Phi, PhiT, R
from safetensors.torch import load_file
from tqdm import tqdm

checkpoint_path = Path('./checkpoint/real/lrdun_3stg.safetensors')

meas_dir_path = Path('./data/real/')
meas_path_list = sorted(list(meas_dir_path.glob('*.mat')))

mask_np = sio.loadmat('./data/mask/real/mask.mat')['mask']
mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).cuda()
Phi_s = Phi(RT(mask_tensor), mask_tensor)

model = LRDUN(stage=3, dim=16, rank=11, bands=28).cuda()
checkpoint = load_file(checkpoint_path)
model.load_state_dict(checkpoint)

save_dir_path = Path('./results/real')
save_dir_path.mkdir(exist_ok=True, parents=True)

preds = []
for meas_path in tqdm(meas_path_list):
    meas_data = sio.loadmat(meas_path)
    meas_tensor = torch.from_numpy(meas_data['meas_real']).float().unsqueeze(0).cuda()
    meas_tensor = meas_tensor / meas_tensor.max() * 0.8
    input = dict(CASSI_measure=meas_tensor, Phi_s=Phi_s, CASSI_mask=mask_tensor)
    with torch.no_grad():
        output = model(input)
    pred = output['pred'].squeeze(0).permute(1,2,0).cpu().numpy()  # [H, W, C]
    save_path = save_dir_path / f'{meas_path.stem}.npy'
    np.save(save_path, pred)
    preds.append(pred)
preds = np.stack(preds, axis=0)  # [N, H, W, C]
sio.savemat(save_dir_path/ 'Test_result.mat', {'pred': preds})
