import sys
import torch
import click
import tntorch as tn
import pytorch_lightning as pl
import numpy as np
from typing import Optional
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
@click.option('-b', '--batch_size', type=int, help='Batch size')
@click.option('-d', '--root', default='/scratch3/MICCAI_BraTS2020_TrainingData', type=str, help='The folder with the dataset')
@click.option('-r', '--results_folder', default='../results', type=str, help='The folder to save results')
@click.option('-e', '--num_epochs', default=100, type=int, help='Number of epochs')
@click.option('-n', '--experiment_name', type=str, help='Name of the experiment')
@click.option('--use_extra_features', type=bool, help='Use additional features to 3d scans')
@click.option('--ngpus', type=int, help='Number of gpus')
@click.option('--pretrain', type=str, default=None, help='Pickle file with pretrain')
@click.option('--only_fold', type=int, help='Number of fold')
def main(
        root: str,
        num_epochs: int,
        results_folder: str,
        batch_size: int,
        experiment_name: str,
        use_extra_features: bool,
        ngpus: int,
        pretrain: Optional[str] = None,
        only_fold: Optional[int] = None):
    sys.path.append('../')

    from utils.dataloaders import BRATS2020
    from models.lightning import ResNetRegresor

    dataset = BRATS2020(root=root)

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
            ResNetRegresor(
                output_dim=1
            ))

        tb_logger = pl.loggers.TensorBoardLogger(
            '../logs/brats_lightning_resnet', name=experiment_name)
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            gpus=ngpus,
            automatic_optimization=False,
            logger=tb_logger,
            resume_from_checkpoint=pretrain)
        trainer.fit(models[-1], dataloaders['train'], dataloaders['val'])


if __name__ == '__main__':
    main()
