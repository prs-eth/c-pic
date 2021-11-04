import sys
import torch
import click
import tntorch as tn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from typing import Optional, Iterable
# torch.autograd.set_detect_anomaly(True)


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
@click.option('--pretrain', type=str, default=None, help='pickle file with pretrain')
@click.option('--resolution', type=int, multiple=True, help='resolution of input')
@click.option('--only_fold', type=int, help='number of fold')
@click.option('--ngpus', type=int, help='number of gpus')
def main(
        dataset_location: str,
        num_epochs: int,
        system_path: str,
        batch_size: int,
        ngpus: int,
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

    from models.lightning import ResNetQuantileRegressor

    print('Resolution {}'.format(resolution))

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
            ResNetQuantileRegressor(
                output_dim=3
            ))

        tb_logger = pl.loggers.TensorBoardLogger('../logs/osic_lightning')
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            gpus=ngpus,
            automatic_optimization=False,
            logger=tb_logger,
            resume_from_checkpoint=pretrain)
        trainer.fit(models[-1], dataloaders['train'], dataloaders['val'])


if __name__ == '__main__':
    main()