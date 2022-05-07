import os
import numpy as np
import torch
from cv2 import cv2
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from pathlib import Path
from PIL import ImageChops, Image

from architectures import fornet
from isplutils import utils, split
from isplutils.data import FrameFaceIterableDataset, load_face
from isplutils.utils import save_model
from isplutils.visualize import vis_attention_on_img

net_name = 'SgeNet'
ckpt_path = 'output/SgeNet-groups8_endpoints/weights/last.pth'

out_path = 'output/att/SgeNet'

img_path = 'input/000_003.mp4-fr000_subj0.jpg'
img_dir = 'input'

net_class: nn.Module = getattr(fornet, net_name)
net = net_class()
# 加载 ckpt
state = torch.load(ckpt_path, map_location='cpu')
net_state = state['net']
incomp_keys = net.load_state_dict(net_state)
print(incomp_keys)
# 加载输入图像
files = os.listdir(img_dir)
for img_name in files:
    img_path = os.path.join(img_dir, img_name)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor: torch.Tensor = torch.from_numpy(img/255.).permute(2,0,1).float()
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    # 预测
    result: tuple = net(img_tensor, debug=True)

    for i, featureMap in enumerate(result):
        if i != 0:
            break
        sub_path = os.path.join(out_path, str(i))
        if not os.path.exists(sub_path):
            make_sub_path = Path(sub_path)
            make_sub_path.mkdir(parents=True)

        # featureMap = F.interpolate(featureMap, (224, 224))
        attention_mask = featureMap.sum(1)
        attention_mask = attention_mask.detach().squeeze().cpu().numpy()
        vis_attention_on_img(img_path, attention_mask, save_path=sub_path)

print('over')
