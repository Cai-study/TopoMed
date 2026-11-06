"""Fixed-split TopoMed training with final checkpoint saving."""
import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from VNet.VNet import VNet
from data.data_loaders import DatasetAllTasks
from data.StrongAug import get_StrongAug
from models_Graph.graph_matching import GModule, set_optimizer, set_scheduler
from utils import poly_lr, read_list, seed_worker, sigmoid_rampup
from utils.torch_compat import autocast, grad_scaler
from utils.config import Config
from utils.config_gcn import config_GCN
from utils.location_scale_augmentation import LocationScaleAugmentation
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--task', required=True)
    parser.add_argument('--exp', default='TopoMed')
    parser.add_argument('--output_dir', default='./logs')
    parser.add_argument('--seed', type=int, help='Run one seed instead of the default three')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 42, 123])
    parser.add_argument('-sl', '--split_labeled', required=True)
    parser.add_argument('-su', '--split_unlabeled', required=True)
    parser.add_argument('--split_test', '-se', '--split_eval', dest='split_test', default='eval',
                        help='Fixed held-out IDs for split validation only')
    parser.add_argument('--data_dir')
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--patch_size', type=int, nargs=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--expected_counts', type=int, nargs=3, metavar=('SOURCE', 'TARGET', 'TEST'))
    parser.add_argument('-ep', '--max_epoch', type=int, default=None,
                        help='Default: 600 for MMWHS, 300 for other tasks')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--sup_loss', choices=['ce', 'wce', 'ce+dice', 'wce+dice', 'w_ce+dice'], default='w_ce+dice')
    parser.add_argument('--unsup_loss', choices=['ce', 'wce', 'ce+dice', 'wce+dice', 'w_ce+dice'], default='w_ce+dice')
    parser.add_argument('-s', '--ema_w', type=float, default=0.99,
                        help='Legacy unused argument; recorded only, no EMA teacher is enabled')
    parser.add_argument('-g', '--gpu', default='0')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('-m', '--mixed_precision', action='store_true', default=True)
    parser.add_argument('--no-mixed_precision', dest='mixed_precision', action='store_false')
    parser.add_argument('--lambda1', '-w', '--mu', dest='lambda1', type=float, default=10)
    parser.add_argument('--lambda2', type=float, default=1)
    parser.add_argument('-r', '--mu_rampup', action='store_true', default=True)
    parser.add_argument('--no-mu_rampup', dest='mu_rampup', action='store_false')
    parser.add_argument('-cr', '--rampup_epoch', type=float)
    parser.add_argument('--confidence_threshold', type=float, default=0.96)
    parser.add_argument('--min_target_pixels', type=int, default=2)
    parser.add_argument('--nodes_per_class', type=int, default=40)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--sinkhorn_iters', type=int, default=20)
    parser.add_argument('--sigma1', type=float, default=0.1)
    parser.add_argument('--sigma2', type=float, default=0.5)
    parser.add_argument('--lda', action='store_true', default=True)
    parser.add_argument('--no-lda', dest='lda', action='store_false')
    parser.add_argument('--gad', action='store_true', default=True)
    parser.add_argument('--no-gad', dest='gad', action='store_false')
    parser.add_argument('--cfgm', action='store_true', default=True)
    parser.add_argument('--no-cfgm', dest='cfgm', action='store_false')
    return parser


def validate_splits(args, config):
    splits = {name: read_list(split, args.task, config=config)
              for name, split in [('source_train', args.split_labeled),
                                  ('target_train', args.split_unlabeled), ('target_test', args.split_test)]}
    for name, ids in splits.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate patient IDs in {name}')
    for train_name in ('source_train', 'target_train'):
        overlap = set(splits[train_name]) & set(splits['target_test'])
        if overlap:
            raise ValueError(f'{train_name}/target_test overlap: {sorted(overlap)}')
    expected = args.expected_counts
    if expected is None:
        if 'mmwhs' in args.task.lower():
            expected = (16, 16, 4)
        elif 'mr2ct' in args.task.lower():
            expected = (16, 24, 6)
        elif 'ct2mr' in args.task.lower():
            expected = (24, 16, 4)
        else:
            raise ValueError('Supply --expected_counts SOURCE TARGET TEST for this task')
    actual = tuple(len(ids) for ids in splits.values())
    if actual != tuple(expected):
        raise ValueError(f'Patient split counts {actual}, expected {tuple(expected)}')
    return splits


def pseudo_targets(logits, threshold=0.96):
    with torch.no_grad():
        confidence, labels = logits.detach().float().softmax(1).max(1, keepdim=True)
        # -1 excludes low-confidence pixels, including uncertain background.
        confident_labels = labels.masked_fill(confidence <= threshold, -1)
    return labels, confident_labels


def Global_LLA_aug(image, target=None, augmenter=None):
    augmenter = augmenter or LocationScaleAugmentation()
    values = image.detach().float().cpu().numpy()
    result = np.stack([augmenter.Global_Location_Scale_Augmentation_target(x, x) for x in values])
    return torch.as_tensor(result, dtype=image.dtype, device=image.device)


def Local_LLA_aug(image, target, image_mask, target_mask, augmenter=None):
    augmenter = augmenter or LocationScaleAugmentation()
    arrays = [x.detach().cpu().numpy() for x in (image, target, image_mask, target_mask)]
    result = np.stack([augmenter.Local_Location_Scale_Augmentation_target(*sample)
                       for sample in zip(*arrays)])
    return torch.as_tensor(result, dtype=image.dtype, device=image.device)


def train_step(model, graph, image_l, label_l, image_u, criterion, augmenter, args, ramp):
    """Real four-view forward/loss spine, also exercised by CPU synthetic checks."""
    sup_criterion, unsup_criterion = criterion if isinstance(criterion, tuple) else (criterion, criterion)
    p_s, _, f_s = model(image_l)
    p_t, _, f_t = model(image_u)
    y_t, confident_y_t = pseudo_targets(p_t, args.confidence_threshold)
    x_s_prime = Local_LLA_aug(image_l, image_u, label_l, confident_y_t, augmenter) if args.lda else image_l
    if args.lda:
        p_sp, _, f_sp = model(x_s_prime)
        loss_sup = sup_criterion(p_s, label_l) + sup_criterion(p_sp, label_l)
    else:
        f_sp = f_s
        loss_sup = sup_criterion(p_s, label_l)
    loss_un = unsup_criterion(p_t, y_t)
    if args.gad:
        x_t_prime = Global_LLA_aug(image_u, augmenter=augmenter)
        p_tp, _, _ = model(x_t_prime)
        y_tp, _ = pseudo_targets(p_tp, args.confidence_threshold)
        loss_un = loss_un + unsup_criterion(p_tp, y_tp)
    graph_losses = {}
    if graph is not None:
        _, _, graph_losses = graph((x_s_prime, image_u), (f_sp, f_t), targets=label_l, score_maps=y_t)
    loss_graph = sum(graph_losses.values(), p_s.new_zeros(()))
    total = loss_sup + args.lambda1 * ramp * loss_un + args.lambda2 * ramp * loss_graph
    parts = {'loss_sup': loss_sup, 'loss_un': loss_un, 'loss_CFGM': loss_graph, **graph_losses}
    return total, parts


def train_one_seed(args, config, splits, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    output = Path(args.output_dir) / args.exp / f'seed_{seed}'
    if (output / 'run_config.json').exists():
        raise FileExistsError(f'Run already exists: {output}; choose another --exp to preserve it')
    (output / 'ckpts').mkdir(parents=True, exist_ok=True)
    with (output / 'run_config.json').open('w', encoding='utf-8') as handle:
        json.dump({'args': vars(args), 'seed': seed, 'config': vars(config), 'splits': splits}, handle, indent=2)
    repeat = max(len(splits['source_train']), len(splits['target_train']), config.batch_size)
    loaders = []
    for index, split in enumerate((args.split_labeled, args.split_unlabeled)):
        dataset = DatasetAllTasks(split, num_cls=config.num_cls, task=args.task, repeat=repeat,
                                  transform=get_StrongAug(config.patch_size, 3, 0.7), unlabeled=bool(index), config=config)
        loaders.append(DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
                                  num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                                  worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed + index)))
    model = VNet(config.num_channels, config.num_cls, config.n_filters,
                 normalization='batchnorm', has_dropout=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9,
                                weight_decay=3e-5, nesterov=True)
    graph = GModule(256, config.num_cls, num_nodes_per_class=args.nodes_per_class,
                    temperature=args.temperature, sinkhorn_iters=args.sinkhorn_iters).to(device) if args.cfgm else None
    graph_optimizer = set_optimizer(graph, config_GCN['gmn']['opt']) if graph is not None else None
    graph_scheduler = set_scheduler(graph_optimizer, config_GCN['gmn']['sch']) if graph is not None else None
    criterion = tuple((RobustCrossEntropyLoss() if name in ('ce', 'wce') else DC_and_CE_loss()).to(device)
                      for name in (args.sup_loss, args.unsup_loss))
    augmenter = LocationScaleAugmentation(min_target_pixels=args.min_target_pixels,
                                         sigma1=args.sigma1, sigma2=args.sigma2)
    amp = args.mixed_precision and device.type == 'cuda'
    scaler = grad_scaler(enabled=amp)
    for epoch in range(args.max_epoch):
        model.train()
        if graph is not None:
            graph.train()
        ramp = sigmoid_rampup(epoch, args.rampup_epoch if args.rampup_epoch is not None else args.max_epoch) if args.mu_rampup else 1.
        for group in optimizer.param_groups:
            group['lr'] = poly_lr(epoch, args.max_epoch, args.base_lr)
        history, steps = {}, 0
        for batch_l, batch_u in zip(*loaders):
            image_l = batch_l['image'].to(device)
            label_l = batch_l['label'].to(device).long().unsqueeze(1)
            image_u = batch_u['image'].to(device)  # never read target annotations
            optimizer.zero_grad(set_to_none=True)
            if graph_optimizer is not None:
                graph_optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=amp):
                loss, parts = train_step(model, graph, image_l, label_l, image_u, criterion, augmenter, args, ramp)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at seed {seed}, epoch {epoch + 1}')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            graph_stepped = graph_optimizer is not None and any(p.grad is not None for p in graph.parameters())
            if graph_stepped:
                scaler.step(graph_optimizer)
            old_scale = scaler.get_scale()
            scaler.update()
            if graph_stepped and scaler.get_scale() >= old_scale:
                graph_scheduler.step()
            for name, value in {'loss': loss, **parts}.items():
                history[name] = history.get(name, 0.) + value.detach().item()
            steps += 1
        record = {'epoch': epoch + 1, 'seed': seed, 'ramp': ramp,
                  **{name: value / steps for name, value in history.items()}}
        with (output / 'train.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        logging.info('%s', record)
    checkpoint = {'state_dict': model.state_dict(), 'epoch': args.max_epoch, 'seed': seed,
                  'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                  'config': vars(config), 'args': vars(args), 'splits': splits}
    if graph is not None:
        checkpoint.update(graph_state_dict=graph.state_dict(), graph_optimizer=graph_optimizer.state_dict(),
                          graph_scheduler=graph_scheduler.state_dict())
    torch.save(checkpoint, output / 'ckpts' / 'final_model.pth')
    return output


def main():
    args = build_parser().parse_args()
    if args.max_epoch is None:
        args.max_epoch = 600 if 'mmwhs' in args.task.lower() else 300
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.max_epoch < 1 or args.batch_size < 1:
        raise ValueError('max_epoch and batch_size must be positive')
    if not 0 <= args.confidence_threshold <= 1 or min(args.sigma1, args.sigma2) < 0:
        raise ValueError('Invalid confidence threshold or Gaussian standard deviation')
    if args.rampup_epoch is not None and args.rampup_epoch < 0:
        raise ValueError('rampup_epoch must be nonnegative')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for small debugging runs')
    config = Config(args.task, args.data_dir, args.num_classes, args.patch_size, args.batch_size)
    if any(size < 16 or size % 16 for size in config.patch_size):
        raise ValueError('VNet patch dimensions must be positive multiples of 16')
    if np.prod([size // 16 for size in config.patch_size]) <= 1:
        raise ValueError('VNet latent InstanceNorm requires more than one spatial voxel')
    splits = validate_splits(args, config)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    seeds = [args.seed] if args.seed is not None else args.seeds
    if len(seeds) != len(set(seeds)):
        raise ValueError('Seeds must be unique')
    for seed in seeds:
        logging.info('Completed run: %s', train_one_seed(args, config, splits, seed))


if __name__ == '__main__':
    main()
