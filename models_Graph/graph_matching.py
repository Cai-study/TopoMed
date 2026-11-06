"""CFGM: class-conditioned graphs and dual-constraint alignment."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from models_Graph.affinity_layer import Affinity
from utils.torch_compat import autocast, isin
from models_Graph.lr_scheduler import WarmupMultiStepLR


class Node_label_weight_mask:
    """Sample at most N_M actual feature vectors per class across all levels.

    The batch is treated as one domain graph, as in the original implementation.
    Only labels are resized; input feature maps are never modified or detached.
    """

    def __init__(self, num_class, num_nodes_per_class=40):
        if num_nodes_per_class < 1:
            raise ValueError('num_nodes_per_class must be positive')
        self.num_class = num_class
        self.num_nodes_per_class = num_nodes_per_class

    def __call__(self, features, mask):
        if torch.is_tensor(features):
            features = [features]
        if not features:
            raise ValueError('At least one feature map is required')
        if mask.ndim == features[0].ndim - 1:
            mask = mask.unsqueeze(1)
        vectors, labels = [], []
        for feature in features:
            resized = F.interpolate(mask.detach().float(), size=feature.shape[2:], mode='nearest').long()
            vectors.append(feature.movedim(1, -1).reshape(-1, feature.shape[1]))
            labels.append(resized.reshape(-1))
        nodes, node_labels = [], []
        # Gather only selected vectors, rather than all class features.
        for cls in range(self.num_class):
            locations = [torch.where(label == cls)[0] for label in labels]
            count = sum(index.numel() for index in locations)
            if count == 0:
                continue
            positions = torch.linspace(0, count - 1, min(count, self.num_nodes_per_class),
                                       device=features[0].device).long()
            offset = 0
            for vector, index in zip(vectors, locations):
                selected = positions[(positions >= offset) & (positions < offset + index.numel())] - offset
                if selected.numel():
                    sampled = vector[index[selected]]
                    nodes.append(sampled)
                    node_labels.append(torch.full((len(sampled),), cls, dtype=torch.long, device=sampled.device))
                offset += index.numel()
        if not nodes:
            empty = features[0].new_empty((0, features[0].shape[1]))
            labels_empty = torch.empty(0, dtype=torch.long, device=empty.device)
            return empty, labels_empty, empty.new_empty(0)
        nodes, node_labels = torch.cat(nodes), torch.cat(node_labels)
        return nodes, node_labels, nodes.new_ones(len(nodes))


class GModule(nn.Module):
    def __init__(self, in_channels, num_classes, device=None, num_nodes_per_class=40,
                 embedding_dim=256, temperature=0.1, sinkhorn_iters=20):
        super().__init__()
        if temperature <= 0 or sinkhorn_iters < 1:
            raise ValueError('temperature and sinkhorn_iters must be positive')
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.graph_generator_3Dmask = Node_label_weight_mask(num_classes, num_nodes_per_class)
        self.head_in_ln = nn.Sequential(
            nn.Linear(in_channels, embedding_dim), nn.LayerNorm(embedding_dim, elementwise_affine=False),
            nn.ReLU(), nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim, elementwise_affine=False))
        self.edge_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.gnn_message = nn.Linear(embedding_dim, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.node_cls_middle = nn.Sequential(nn.Linear(embedding_dim, embedding_dim * 2),
                                            nn.ReLU(), nn.Linear(embedding_dim * 2, num_classes))
        self.node_affinity = Affinity(d=embedding_dim)

    def _forward_intra_domain_graph(self, nodes):
        projected = self.edge_projection(nodes)
        edges = (projected @ projected.T).softmax(dim=-1)
        return nodes + F.relu(self.gnn_message(edges @ nodes)), edges

    def _forward_cross_domain_graph(self, source, target):
        scale = math.sqrt(source.shape[-1])
        # Both directions read the same pre-interaction features.
        st = (self.query(source) @ self.key(target).T / scale).softmax(-1)
        ts = (self.query(target) @ self.key(source).T / scale).softmax(-1)
        return source + st @ self.value(target), target + ts @ self.value(source)

    def _semantic_loss(self, affinity, source_labels, target_labels):
        positive = source_labels[:, None] == target_labels[None, :]
        scores = affinity / self.temperature
        # Filtering to common classes guarantees a positive for every row.
        return (torch.logsumexp(scores, dim=1)
                - torch.logsumexp(scores.masked_fill(~positive, -float('inf')), dim=1)).mean()

    @staticmethod
    def sinkhorn_rpm(log_alpha, n_iters=20, slack=True, eps=-1):
        """Log-space Sinkhorn; slack row/column are excluded from normalization."""
        if slack:
            padded = F.pad(log_alpha, (0, 1, 0, 1), value=0)
            for _ in range(n_iters):
                rows = padded[..., :-1, :] - torch.logsumexp(padded[..., :-1, :], dim=-1, keepdim=True)
                padded = torch.cat((rows, padded[..., -1:, :]), dim=-2)
                cols = padded[..., :, :-1] - torch.logsumexp(padded[..., :, :-1], dim=-2, keepdim=True)
                padded = torch.cat((cols, padded[..., :, -1:]), dim=-1)
            return padded[..., :-1, :-1]
        for _ in range(n_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return log_alpha

    def _normalized_affinity(self, affinity):
        # Equivalent to non-affine IN per graph; defined for a single element too.
        centered = affinity - affinity.mean()
        normalized = centered * torch.rsqrt(centered.square().mean() + 1e-5)
        return self.sinkhorn_rpm(normalized, self.sinkhorn_iters).exp()

    @staticmethod
    def _forward_qu(edge_1, edge_2, affinity):
        return (edge_1 @ affinity - affinity @ edge_2).abs().mean()

    def forward(self, images, features, targets=None, score_maps=None):
        if targets is None:
            return features, None
        if score_maps is None:
            raise ValueError('Target pseudo labels are required for CFGM training')
        source, sl, _ = self.graph_generator_3Dmask(features[0], targets)
        target, tl, _ = self.graph_generator_3Dmask(features[1], score_maps)
        if len(source) == 0 or len(target) == 0:
            return features, (source, target), {}
        # Matrix products and Sinkhorn run in FP32 even under outer AMP.
        with autocast(device_type=source.device.type, enabled=False):
            source, se = self._forward_intra_domain_graph(self.head_in_ln(source.float()))
            target, te = self._forward_intra_domain_graph(self.head_in_ln(target.float()))
            source, target = self._forward_cross_domain_graph(source, target)
            losses = {'loss_ND': F.cross_entropy(self.node_cls_middle(source), sl)
                      + F.cross_entropy(self.node_cls_middle(target), tl)}
            source_classes, target_classes = sl.unique(), tl.unique()
            common = source_classes[isin(source_classes, target_classes)]
            if common.numel():
                si = torch.where(isin(sl, common))[0]
                ti = torch.where(isin(tl, common))[0]
                affinity = self.node_affinity(source[si], target[ti])
                losses['loss_SC'] = self._semantic_loss(affinity, sl[si], tl[ti])
                matched = self._normalized_affinity(affinity)
                # Induced subgraphs: retain original edges without renormalizing.
                losses['loss_TC'] = self._forward_qu(se[si][:, si], te[ti][:, ti], matched)
        return features, (source, target), losses


def set_optimizer(net, config_GCN):
    if config_GCN['opt_name'] == 'SGD':
        return torch.optim.SGD(net.parameters(), lr=config_GCN['lr'],
                               weight_decay=config_GCN['weight_decay'], momentum=config_GCN['momentum'])
    if config_GCN['opt_name'] == 'Adam':
        return torch.optim.Adam(net.parameters(), lr=config_GCN['lr'], weight_decay=config_GCN['weight_decay'])
    raise ValueError(config_GCN['opt_name'])


def set_scheduler(optimizer, config_GCN):
    return WarmupMultiStepLR(optimizer, config_GCN['STEPS'], config_GCN['GAMMA'],
                            warmup_factor=config_GCN['WARMUP_FACTOR'],
                            warmup_iters=config_GCN['WARMUP_ITERS'], warmup_method=config_GCN['WARMUP_METHOD'])
