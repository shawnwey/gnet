from pprint import pprint
from typing import Iterable, List

import albumentations as A
import cv2
import numpy as np
import scipy
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import ToPILImage, ToTensor
import pandas as pd
from matplotlib import pyplot as plt
from torch import nn as nn
from sklearn.metrics import roc_auc_score
from PIL import ImageChops, Image
from tqdm import tqdm
from tensorboardX import SummaryWriter

from isplutils.data import FrameFaceIterableDataset, load_face


def validation(net, device, val_loader, criterion, tb, iteration, tag: str, loader_len_norm: int = None):
    net.eval()
    loader_len_norm = loader_len_norm if loader_len_norm is not None else val_loader.batch_size
    val_num = 0
    val_loss = 0.
    pred_list = list()
    labels_list = list()
    for val_data in tqdm(val_loader, desc='Val', leave=True):
        batch_data, batch_labels = val_data

        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(net, device, criterion, batch_data,
                                                           batch_labels)
        pred_list.append(val_batch_pred.flatten())
        val_num += val_batch_num
        val_loss += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss /= val_num
    tb.add_scalar('{}/loss'.format(tag), val_loss, iteration)

    val_roc_auc = 0
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        val_labels = np.concatenate(labels_list)
        val_pred = np.concatenate(pred_list)
        val_roc_auc = roc_auc_score(val_labels, val_pred)
        tb.add_scalar('{}/roc_auc'.format(tag), val_roc_auc, iteration)
        tb.add_pr_curve('{}/pr'.format(tag), val_labels, val_pred, iteration)

    return val_loss, val_roc_auc

def batch_forward(net: nn.Module, device: torch.device, criterion, data: torch.Tensor, labels: torch.Tensor) -> (
        torch.Tensor, float, int):
    data = data.to(device)
    labels = labels.to(device)
    out = net(data)
    pred = torch.sigmoid(out).detach().cpu().numpy()
    loss = criterion(out, labels)
    return loss, pred


def tb_attention(tb: SummaryWriter,
                 tag: str,
                 iteration: int,
                 net: nn.Module,
                 device: torch.device,
                 patch_size_load: int,
                 face_crop_scale: str,
                 val_transformer: A.BasicTransform,
                 root: str,
                 record: pd.Series,
                 ):
    # Crop face
    sample_t = load_face(record=record, root=root, size=patch_size_load, scale=face_crop_scale,
                         transformer=val_transformer)
    sample_t_clean = load_face(record=record, root=root, size=patch_size_load, scale=face_crop_scale,
                               transformer=ToTensorV2())
    if torch.cuda.is_available():
        sample_t = sample_t.cuda(device)
    # Transform
    # Feed to net
    with torch.no_grad():
        att: torch.Tensor = net.get_attention(sample_t.unsqueeze(0))[0].cpu()
    att_img: Image.Image = ToPILImage()(att)
    sample_img = ToPILImage()(sample_t_clean)
    att_img = att_img.resize(sample_img.size, resample=Image.NEAREST).convert('RGB')
    sample_att_img = ImageChops.multiply(sample_img, att_img)
    sample_att = ToTensor()(sample_att_img)
    tb.add_image(tag=tag, img_tensor=sample_att, global_step=iteration)