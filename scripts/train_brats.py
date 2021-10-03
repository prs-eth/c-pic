import sys
import torch
import click
import pytorch_lightning as pl
import numpy as np
from typing import Optional
torch.set_default_dtype(torch.float64)
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
@click.option('--use_encoder', type=bool, help='Go without encoder ?')
@click.option('-d', '--root', default='/scratch3/MICCAI_BraTS2020_TrainingData', type=str, help='The folder with the dataset')
@click.option('-r', '--results_folder', default='../results', type=str, help='The folder to save results')
@click.option('-e', '--num_epochs', default=100, type=int, help='Number of epochs')
@click.option('-n', '--experiment_name', type=str, help='Name of the experiment')
@click.option('--use_extra_features', type=bool, help='Use additional features to 3d scans')
@click.option('--pretrain', type=str, default=None, help='Pickle file with pretrain')
@click.option('--only_fold', type=int, help='Number of fold')
@click.option('--ndims', type=int, help='Number of encoder dims')
@click.option('--ngpus', type=int, help='Number of gpus')
@click.option('--rank', type=int, help='Max TT rank')
@click.option('--num_workers', type=int, help='Number of workers for dataloaders')
def main(
        root: str,
        num_epochs: int,
        results_folder: str,
        batch_size: int,
        use_encoder: bool,
        experiment_name: str,
        use_extra_features: bool,
        ngpus: int,
        rank: int,
        ndims: Optional[int] = None,
        pretrain: Optional[str] = None,
        only_fold: Optional[int] = None,
        num_workers: Optional[int] = 2):
    sys.path.append('../')

    from utils.dataloaders import BRATS2020
    from utils.accessors import build_enc_acc3d, build_acc3d_multidim, build_no_enc_acc3d
    from utils.helpers_3d import get_3dtensors_patches
    from models.lightning import TTRegressor

    res = [256, 256, 256]
    if not use_encoder:
        build_enc_f = build_no_enc_acc3d
        new_shape = res
    else:
        if ndims in [None, 1]:
            ndims = 1
            build_enc_f = build_enc_acc3d
            new_shape = res
        else:
            build_enc_f = build_acc3d_multidim
            new_shape = res + [ndims]

    print('ndims encoder = {}'.format(ndims))

    q_shape = [2] * int((np.log(new_shape) / np.log(2)).sum())

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
                shuffle=True,
                num_workers=num_workers,
                drop_last=True),
            'val': torch.utils.data.DataLoader(
                valset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True)}

        models.append(
            TTRegressor(
                build_encoder_function=build_enc_f,
                path_function=get_3dtensors_patches,
                use_encoder=use_encoder,
                q_shape=q_shape,
                ndims=ndims,
                rank=rank,
                full_shape=new_shape,
                use_extra_features=False))

        tb_logger = pl.loggers.TensorBoardLogger(
            '../logs/brats_lightning', name=experiment_name)
        trainer = pl.Trainer(   
            max_epochs=num_epochs,
            gpus=ngpus,
            logger=tb_logger,
            resume_from_checkpoint=pretrain,
            num_processes=1)
        trainer.fit(models[-1], dataloaders['train'], dataloaders['val'])


if __name__ == '__main__':
    main()
