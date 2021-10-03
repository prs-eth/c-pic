import torch


def compute_MSE(orig, pred):
    return torch.nn.MSELoss()(pred.reshape(-1), orig.reshape(-1))


def compute_CE(orig, pred):
    return torch.nn.functional.cross_entropy(pred, orig)