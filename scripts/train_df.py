import os
import sys
import time
import torch
import click
import pickle
import tntorch as tn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from functools import partial
from typing import Optional, Iterable
from sklearn.model_selection import train_test_split
torch.autograd.set_detect_anomaly(True)


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)


@click.command()
@click.option('-b', '--batch_size', type=int, help='Batch size')
@click.option('-d', '--root', default='/scratch3/MICCAI_BraTS2020_TrainingData', type=str, help='The folder with the dataset')
@click.option('-r', '--results_folder', default='../results', type=str, help='The folder to save results')
@click.option('-e', '--num_epochs', default=100, type=int, help='Number of epochs')
@click.option('-n', '--experiment_name', type=str, help='Name of the experiment')
@click.option('--use_extra_features', type=bool, help='Use additional features to 3d scans')
@click.option('--pretrain', type=str, default=None, help='pickle file with pretrain')
@click.option('--only_fold', type=int, help='number of fold')
@click.option('--ndims', type=int, help='number of encoder dims')
@click.option('--ngpus', type=int, help='number of gpus')
def main(
        root: str,
        num_epochs: int,
        results_folder: str,
        batch_size: int,
        experiment_name: str,
        use_extra_features: bool,
        ngpus: int,
        ndims: Optional[int] = None,
        pretrain: Optional[str] = None,
        only_fold: Optional[int] = None):
    sys.path.append('../')

    from utils.dataloaders import DFDataset

    from utils.accessors import build_enc_acc3d, build_acc3d_multidim
    from utils.helpers_3d import get_3dtensors_patches
    from models.lightning import TTRegressor, TTClassifier, add_dim

    if ndims in [None, 1]:
        ndims = 1
        build_enc_f = build_enc_acc3d
        new_shape = (16, 16, 16)
    else:
        build_enc_f = build_acc3d_multidim
        new_shape = (16, 16, 16, ndims)

    print('ndims encoder = {}'.format(ndims))

    q_shape = [2] * int((np.log(new_shape) / np.log(2)).sum())
    rank = 10

    dataset = DFDataset(root=root, resolution=16)

    models = []
    folds = dataset.kfold(5)
    for fold, (trainset, valset) in enumerate(folds):
        if only_fold is not None and fold != only_fold:
            continue
        print('Fold {}'.format(fold))

        dataloaders = {
            'train': torch.utils.data.DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=True, num_workers=2),
            'val': torch.utils.data.DataLoader(
                valset,
                batch_size=batch_size,
                shuffle=False, num_workers=2)}

        models.append(
            TTClassifier(
                build_encoder_function=build_enc_f,
                path_function=get_3dtensors_patches,
                q_shape=q_shape,
                ndims=ndims,
                rank=rank,
                full_shape=new_shape,
                num_classes=len(dataset.categories),
                resolution=[16, 16, 16],
                arg_patch_function=add_dim))

        tb_logger = pl.loggers.TensorBoardLogger('../logs/df_lightning')
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            gpus=ngpus,
            automatic_optimization=False,
            logger=tb_logger,
            resume_from_checkpoint=pretrain)
        trainer.fit(models[-1], dataloaders['train'], dataloaders['val'])


if __name__ == '__main__':
    main()
