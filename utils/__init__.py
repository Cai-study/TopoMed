import os
import math
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.config import Config
import math




def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def EMA(cur_weight, past_weight, momentum=0.9):
    new_weight = momentum * past_weight + (1 - momentum) * cur_weight
    return new_weight


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def print_func(item):
    if type(item) == torch.Tensor:
        return [round(x, 2) for x in item.data.cpu().numpy().tolist()]
    elif type(item) == np.ndarray:
        return [round(x, 2) for x in item.tolist()]
    else:
        raise TypeError


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp / (x_sum)
    return s


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent


def maybe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_nifti(path):
    import SimpleITK as sitk
    itk_img = sitk.ReadImage(path)

    itk_arr = sitk.GetArrayFromImage(itk_img)
    return itk_arr


def read_list(split, task, config=None):
    config = config if config is not None else Config(task)
    ids_list = np.atleast_1d(np.loadtxt(
        os.path.join(config.save_dir, 'split', f'{split}.txt'),
        dtype=str
    )).tolist()
    if not ids_list:
        raise ValueError(f'Empty split: {split}')
    return sorted(ids_list)


def read_data(data_id, task, normalize=False, config=None, unlabeled=False):
    config = config if config is not None else Config(task)
    im_path = os.path.join(config.save_dir, 'npy', f'{data_id}_image.npy')
    lb_path = os.path.join(config.save_dir, 'npy', f'{data_id}_label.npy')
    if not os.path.exists(im_path) or (not unlabeled and not os.path.exists(lb_path)):
        print(im_path)
        print(lb_path)
        raise ValueError(data_id)
    image = np.load(im_path)
    label = np.zeros(image.shape, dtype=np.int16) if unlabeled else np.load(lb_path)

    if normalize:
        if "synapse" in task:
            image = image.clip(min=-75, max=275)
        elif "mnms" in task:
            p5 = np.percentile(image.flatten(), 0.5)
            p95 = np.percentile(image.flatten(), 99.5)
            image = image.clip(min=p5, max=p95)

        image = (image - image.min()) / max(float(np.ptp(image)), 1e-8)

        image = image.astype(np.float32)

    return image, label


def read_list_2d(split, task):
    config = Config(task + "_2d")
    ids_list = np.loadtxt(
        os.path.join(config.save_dir, 'split', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)


def read_data_2d(data_id, task, normalize=False):
    import cv2
    import h5py
    config = Config(task + "_2d")
    if "acdc" in task:
        h5File = h5py.File(os.path.join(config.save_dir, 'h5', f'{data_id}.h5'), 'r')
        image = h5File["image"][:]
        label = h5File["label"][:]
    else:
        im_path = os.path.join(config.save_dir, 'png', f'{data_id}_image.png')
        lb_path = os.path.join(config.save_dir, 'png', f'{data_id}_label.png')
        if not os.path.exists(im_path) or not os.path.exists(lb_path):
            print(im_path)
            print(lb_path)
            raise ValueError(data_id)
        image = cv2.imread(im_path, 0)
        label = cv2.imread(lb_path, 0)

    if normalize:
        if "synapse" in task:
            image = image.clip(min=-75, max=275)
        elif "mnms" in task:
            p5 = np.percentile(image.flatten(), 0.5)
            p95 = np.percentile(image.flatten(), 99.5)
            image = image.clip(min=p5, max=p95)

        image = (image - image.min()) / (image.max() - image.min())

        image = image.astype(np.float32)

    return image, label


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fetch_data(batch, labeled=True):
    image = batch['image'].cuda()
    if labeled:
        label = batch['label'].cuda().unsqueeze(1)
        return image, label
    else:
        return image
