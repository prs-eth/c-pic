import sys
import torch
import click
import numpy as np
import pytorch_lightning as pl
from typing import Optional, Iterable
torch.set_default_dtype(torch.float64)
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
@click.option('-d', '--dataset', type=str, help='Dataset')
@click.option('-r', '--results_folder', default='../results', type=str, help='The folder to save results')
@click.option('-e', '--num_epochs', default=100, type=int, help='Number of epochs')
@click.option('-n', '--experiment_name', type=str, help='Name of the experiment')
@click.option('--pretrain', type=str, default=None, help='pickle file with pretrain')
@click.option('--only_fold', type=int, help='number of fold')
@click.option('--ndims', type=int, help='number of encoder dims')
@click.option('--resolution', type=int, multiple=True, help='Resolution')
@click.option('--num_classes', type=int, help='Number of classes')
@click.option('--num_examples', type=int, help='Number of examples')
@click.option('--df', type=bool, default=False, help='use df or not')
@click.option('--ngpus', type=int, help='number of gpus')
def main(
        dataset: str,
        num_epochs: int,
        results_folder: str,
        batch_size: int,
        experiment_name: str,
        num_classes: int,
        num_examples: int,
        resolution: Iterable[int],
        ngpus: int,
        ndims: Optional[int] = None,
        pretrain: Optional[str] = None,
        only_fold: Optional[int] = None,
        df: Optional[bool] = False):
    sys.path.append('../')
    sys.path.append('/scratch/lib')

    from utils.dataloaders import MNIST3D, ModelNet
    from utils.accessors import build_enc_acc3d, build_acc3d_multidim
    from utils.helpers_3d import get_3dtensors_patches, get_nndf_patches, get_df_patches
    from models.lightning import TTClassifier, TTRegressor, add_dim

    if ndims in [None, 1]:
        ndims = 1
        build_enc_f = build_enc_acc3d
        new_shape = resolution
    else:
        build_enc_f = build_acc3d_multidim
        new_shape = list(resolution) + [ndims]

    print('Resolution {}'.format(resolution))
    print('ndims encoder = {}'.format(ndims))
    
    q_shape = [2] * int((np.log(new_shape) / np.log(2)).sum())
    rank = 10

    # NOTE(aelphy): mind that resolution is the same
    if dataset == 'MNIST':
        ds = MNIST3D(folder='../data/3d-mnist', resolution=resolution[0], num_examples=num_examples, df=df)
    elif dataset == 'ModelNet':
        ds = ModelNet(root='../data/ModelNet10', name='10', res=resolution[0])
    else:
        raise RuntimeError('Dataset is not supported')

    models = []
    folds = ds.kfold(5)

    for fold, (trainset, valset) in enumerate(folds):
        if only_fold is not None and fold != only_fold:
            continue
        print('Fold {}'.format(fold))

        if dataset == 'MNIST':
            dataloaders = {
                'train': torch.utils.data.DataLoader(
                    trainset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2,
                    drop_last=True),
                'val': torch.utils.data.DataLoader(
                    valset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,
                    drop_last=True)}
        else:
            dataloaders = {
                'train': torch.utils.data.DataLoader(
                    trainset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2,
                    collate_fn=ds.collate_wrapper,
                    drop_last=True),
                'val': torch.utils.data.DataLoader(
                    valset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,
                    collate_fn=ds.collate_wrapper,
                    drop_last=True)}

        if dataset == 'MNIST':
            models.append(
                TTClassifier(
                    resolution=resolution[0],
                    build_encoder_function=build_enc_f,
                    path_function=get_3dtensors_patches,
                    q_shape=q_shape,
                    ndims=ndims,
                    rank=rank,
                    num_classes=num_classes,
                    full_shape=new_shape,
                    arg_patch_function=add_dim))
        else:
            models.append(
                TTClassifier(
                    resolution=resolution[0],
                    build_encoder_function=build_enc_f,
                    path_function=get_df_patches,
                    q_shape=q_shape,
                    ndims=ndims,
                    rank=rank,
                    num_classes=num_classes,
                    full_shape=new_shape))

        tb_logger = pl.loggers.TensorBoardLogger('../logs/{}_lightning'.format(dataset))
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            gpus=ngpus,
            automatic_optimization=False,
            logger=tb_logger,
            resume_from_checkpoint=pretrain)
        trainer.fit(models[-1], dataloaders['train'], dataloaders['val'])


if __name__ == '__main__':
    main()
