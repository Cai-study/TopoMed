"""AMP dispatch for both current and legacy CUDA PyTorch APIs."""
from contextlib import nullcontext
import torch


def autocast(device_type, enabled=True):
    if hasattr(torch, 'autocast'):
        return torch.autocast(device_type=device_type, enabled=enabled)
    if device_type == 'cuda':
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def grad_scaler(enabled=True):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def isin(elements, candidates):
    return (elements[..., None] == candidates).any(dim=-1)


