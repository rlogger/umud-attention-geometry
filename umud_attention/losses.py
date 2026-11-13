"""Masked overlap and differentiable skeleton consistency for thin fascicles."""
import torch
from torch.nn import functional as F


def soft_erode(x):
    return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)),
                         -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def soft_skeleton(x, iterations=8):
    opened = F.max_pool2d(soft_erode(x), 3, 1, 1)
    skeleton = F.relu(x - opened)
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - F.max_pool2d(soft_erode(x), 3, 1, 1))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def dice_per_image(prob, target, valid):
    axes = (1, 2, 3)
    p, y = prob * valid, target * valid
    return (2 * (p * y).sum(axes) + 1e-6) / (p.sum(axes) + y.sum(axes) + 1e-6)


def segmentation_loss(logits, target, valid, topology_weight=0.15):
    logits, target, valid = logits.float(), target.float(), valid.float()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    p, y = logits.sigmoid() * valid, target * valid
    dice = 1 - dice_per_image(p, y, valid).mean()
    topology = p.sum() * 0
    if topology_weight:
        sp, sy = soft_skeleton(p), soft_skeleton(y)
        axes = (1, 2, 3)
        precision = ((sp * y).sum(axes) + 1e-6) / (sp.sum(axes) + 1e-6)
        sensitivity = ((sy * p).sum(axes) + 1e-6) / (sy.sum(axes) + 1e-6)
        topology = (1 - (2 * precision * sensitivity + 1e-6) / (precision + sensitivity + 1e-6)).mean()
    return 0.4 * bce + 0.6 * dice + topology_weight * topology
