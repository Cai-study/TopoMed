import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default='mmwhs_ct2mr')
parser.add_argument('--exp', type=str, default='SSDA_ckpts/fold1')
parser.add_argument('--split', type=str, default='test_demo')
parser.add_argument('--speed', type=int, default=0)
parser.add_argument('-g', '--gpu', type=str,  default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from model.model import vnet
from utils import read_list, maybe_mkdir, test
from utils import config

config = config.Config(args.task)

if __name__ == '__main__':
    stride_dict = {
        0: (16, 4),
        1: (64, 16),
        2: (128, 32),
    }
    stride = stride_dict[args.speed]

    snapshot_path = f'./logs/{args.exp}/'
    test_save_path = f'./logs/{args.exp}/predictions/'
    maybe_mkdir(test_save_path)
    print(snapshot_path)

    model = vnet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=False
    ).cuda()

    ckpt_path = os.path.join(snapshot_path, f'ckpts/model.pth')
    with (torch.no_grad()):
        model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
        model.eval()
        print(f'load checkpoint from {ckpt_path}')
        test(
            args.task,
            model,
            read_list(args.split, task=args.task),
            num_classes=config.num_cls,
            patch_size=config.patch_size,
            stride_xy=stride[0],
            stride_z=stride[1],
            test_save_path=test_save_path
        )