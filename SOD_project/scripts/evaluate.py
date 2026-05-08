import torch

def iou(pred, target):
    pred = (pred > 0.4).float()
    target = target.float()
    intersection = (pred * target).sum()
    union = (pred + target - pred * target).sum()
    return (intersection + 1e-6) / (union + 1e-6)

def precision(pred, target):
    pred = (pred > 0.4).float()
    target = target.float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + 1e-6) / (tp + fp + 1e-6)

def recall(pred, target):
    pred = (pred > 0.4).float()
    target = target.float()
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + 1e-6) / (tp + fn + 1e-6)

def f1(pred, target):
    p = precision(pred, target)
    r = recall(pred, target)
    return (2 * p * r + 1e-6) / (p + r + 1e-6)