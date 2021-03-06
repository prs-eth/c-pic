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
@click.option('-s', '--system_path', default='/cluster/work/igp_psr/mikhailu/lib', multiple=True, type=str, help='Path to add to PYTHON_PATH')
@click.option('-b', '--batch_size', type=int, help='Batch size')
@click.option('-d', '--dataset_location', default='../data/osic/', type=str, help='The folder with the dataset')
@click.option('-e', '--num_epochs', default=100, type=int, help='Number of epochs')
@click.option('--use_encoder', type=bool, help='Go without encoder ?')
@click.option('--pretrain', type=str, default=None, help='pickle file with pretrain')
@click.option('--resolution', type=int, multiple=True, help='resolution of input')
@click.option('--only_fold', type=int, help='number of fold')
@click.option('--ndims', type=int, help='number of encoder dims')
@click.option('--ngpus', type=int, help='number of gpus')
def main(
        dataset_location: str,
        num_epochs: int,
        system_path: str,
        use_encoder: bool,
        batch_size: int,
        ngpus: int,
        ndims: Optional[int] = None,
        pretrain: Optional[str] = None,
        resolution: Optional[Iterable[int]] = None,
        only_fold: Optional[int] = None):
    sys.path.append('../')
    sys.path.append('/scratch/lib')
    sys.path.append('../../lib')
    sys.path.append('../../../lib')

    for path in system_path:
        sys.path.append(path)

    from utils.dataloaders import Dicom

    train_df = pd.read_csv(os.path.join(dataset_location, 'train.csv'))
    train_patients, validation_patients = train_test_split(train_df.Patient.unique())

    from utils.accessors import build_enc_acc3d, build_acc3d_multidim, build_no_enc_acc3d
    from utils.helpers_3d import get_3dtensors_patches
    from models.lightning import TTQuantileRegressor

    quantiles = (0.2, 0.5, 0.8)

    if not use_encoder:
        build_enc_f = build_no_enc_acc3d
        new_shape = (32, 512, 512)
    else:
        if ndims in [None, 1]:
            ndims = 1
            build_enc_f = build_enc_acc3d
            new_shape = resolution or (32, 512, 512)
        else:
            build_enc_f = build_acc3d_multidim
            new_shape = list(resolution) + [ndims] or (32, 512, 512, ndims)

    print('Resolution {}'.format(resolution))
    print('ndims encoder = {}'.format(ndims))

    q_shape = [2] * int((np.log(new_shape) / np.log(2)).sum())
    rank = 10

    dataset = Dicom(
        path=dataset_location,
        features=None,
        train_patients=train_patients,
        resolution=resolution)

    models = []
    folds = dataset.group_kfold(5)
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
            TTQuantileRegressor(
                resolution=resolution[0],
                build_encoder_function=build_enc_f,
                use_encoder=use_encoder,
                path_function=get_3dtensors_patches,
                q_shape=q_shape,
                ndims=ndims,
                rank=rank,
                full_shape=new_shape,
                use_extra_features=False,
                quantiles=quantiles))

        tb_logger = pl.loggers.TensorBoardLogger('../logs/osic_lightning')
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            gpus=ngpus,
            logger=tb_logger,
            resume_from_checkpoint=pretrain)
        trainer.fit(models[-1], dataloaders['train'], dataloaders['val'])


if __name__ == '__main__':
    main()
