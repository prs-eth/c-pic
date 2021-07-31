import os
import sys
import time
import torch
import click
import pickle
import tntorch as tn
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


def quantile_loss(preds, target, quantiles):
    assert not target.requires_grad
    assert preds.size(0) == target.size(0)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss


def metric(preds, targets):
    sigma = preds[:, 2] - preds[:, 0]
    sigma[sigma < 70] = 70
    delta = (preds[:, 1] - targets).abs()
    delta[delta > 1000] = 1000
    return -np.sqrt(2) * delta / sigma - torch.log(np.sqrt(2) * sigma)


@click.command()
@click.option('-f', '--features', type=click.Choice(['CNN', 'TT', 'TTE2E']), help='Type of features for 3D scans')
@click.option('--tt_pkl_path', type=str, help='Path to saved TT of inputs')
@click.option('-s', '--system_path', default='/cluster/work/igp_psr/mikhailu/lib', multiple=True, type=str, help='Path to add to PYTHON_PATH')
@click.option('-b', '--batch_size', type=int, help='Batch size')
@click.option('-d', '--dataset_location', default='../data/osic/', type=str, help='The folder with the dataset')
@click.option('-r', '--results_folder', default='../results', type=str, help='The folder to save results')
@click.option('-e', '--num_epochs', default=100, type=int, help='Number of epochs')
@click.option('-n', '--experiment_name', type=str, help='Name of the experiment')
@click.option('--use_extra_features', type=bool, help='Use additional features to 3d scans')
@click.option('--eval_every', type=int, help='Evaluate every N epochs')
@click.option('--pretrain', type=str, default=None, help='pickle file with pretrain')
@click.option('--resolution', type=int, multiple=True, help='resolution of input')
@click.option('--only_fold', type=int, help='number of fold')
@click.option('--ndims', type=int, help='number of encoder dims')
def main(
        dataset_location: str,
        num_epochs: int,
        results_folder: str,
        system_path: str,
        features: str,
        tt_pkl_path: str,
        batch_size: int,
        experiment_name: str,
        eval_every: int,
        use_extra_features: bool,
        ndims: Optional[int] = None,
        pretrain: Optional[str] = None,
        resolution: Optional[Iterable[int]] = None,
        only_fold: Optional[int] = None):
    sys.path.append('../')

    for path in system_path:
        sys.path.append(path)

    from utils.radam import RAdam
    from utils.dataloaders import Dicom
    from utils.tt_approx import merge_cores_to_batch, get_train_features
    from models.dense import TwoHeadsRegressor
    from models.resnet3d import generate_model

    train_df = pd.read_csv(os.path.join(dataset_location, 'train.csv'))
    train_patients, validation_patients = train_test_split(train_df.Patient.unique())

    if features == 'CNN':
        dataset = Dicom(path=dataset_location, features=None, train_patients=train_patients, resolution=resolution)
    elif features == 'TT':
        with open(tt_pkl_path, 'rb') as f:
            recs = pickle.load(f)[0]
            rank = recs[0].ranks_tt.max()
            features_train, _ = get_train_features(merge_cores_to_batch(recs), rank)
        
        dataset = Dicom(path=dataset_location, features=features_train, train_patients=train_patients, resolution=resolution)
    elif features == 'TTE2E':
        from models.cnn import EncoderCA_3D, DecoderCA_3D
        from utils.accessors import build_enc_acc3d, build_acc3d_multidim
        from utils.svd import truncate_ca_qtt
        from utils.tt_approx import get_test_features
        from utils.helpers_3d import get_3dtensors_patches

        if ndims in [None, 1]:
            ndims = 1
            build_enc_f = build_enc_acc3d
            new_shape = resolution or (32, 512, 512)
        else:
            build_enc_f = partial(build_acc3d_multidim, arg_patch_func=lambda x: x[None, ...].cpu(), patch_func=get_3dtensors_patches)
            new_shape = (list(resolution) + [ndims]) or (32, 512, 512, ndims)

        print('ndims encoder = {}'.format(ndims))
        
        q_shape = [2] * int((np.log(new_shape) / np.log(2)).sum())
        rank = 10
    
        dataset = Dicom(path=dataset_location, features=None, train_patients=train_patients, resolution=resolution)

    quantiles = (0.2, 0.5, 0.8)
    models = []
    resnets = []
    folds = dataset.group_kfold(5)
    if features == 'TTE2E':
        encoders = []

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

        if features == 'CNN':
            resnet = generate_model(34, n_classes=0, n_input_channels=1).cuda()

            if use_extra_features:
                regressor = TwoHeadsRegressor(8, 512 + 100, 3).cuda()
            else:
                regressor = TwoHeadsRegressor(1, 512 + 100, 3).cuda()
            optimizer = RAdam(list(resnet.parameters()) + list(regressor.parameters()), lr=1e-3)

            resnets.append(resnet)
        elif features == 'TT':
            if use_extra_features:
                regressor = TwoHeadsRegressor(8, rank + 100, 3).cuda()
            else:
                regressor = TwoHeadsRegressor(1, rank + 100, 3).cuda()
            optimizer = RAdam(list(regressor.parameters()), lr=1e-3)
        elif features == 'TTE2E':
            if use_extra_features:
                regressor = TwoHeadsRegressor(8, rank + 100, 3).cuda()
            else:
                regressor = TwoHeadsRegressor(1, rank + 100, 3).cuda()

            # NOTE(aelphy): mind the batch size here.
            enc = EncoderCA_3D(num_out_channels=ndims).cuda()
            encoder = partial(build_enc_f, enc=enc, q_shape=q_shape, full_shape=new_shape, detach=False, batch_size=10000)
            optimizer = RAdam(list(enc.parameters()) + list(regressor.parameters()), lr=1e-3)

            encoders.append(enc)

            if pretrain is not None:
                with open(pretrain, 'rb') as f:
                    data = pickle.load(f)

                encoders[-1].load_state_dict(data[0])
                regressor.load_state_dict(data[1])

        models.append(regressor)
        failed_states = []

        for j in range(num_epochs):
            t0 = time.time()
            counter = 0
            counter_val = 0
            loss = 0
            loss_val = 0
            m = 0
            m_val = 0

            models[-1].train()
            if features == 'TTE2E':
                encoders[-1].train()
            qtts = []
            for batch_index, (tensor, other_features, labels, init_values) in enumerate(dataloaders['train']):
                optimizer.zero_grad()

                print('Batch {}'.format(batch_index))

                if features == 'CNN':
                    resnets[-1].train()
                    cnn_features = resnets[-1](tensor[:, None, ...].cuda())
                    preds = models[-1](cnn_features, other_features.cuda())
                elif features == 'TT':
                    preds = models[-1](tensor.cuda(), other_features.cuda())
                elif features == 'TTE2E':
                    accs = [None] * len(tensor)
                    for i, im in enumerate(tensor):
                        accs[i] = encoder(im=im)

                    trial = 0
                    failed_example = False
                    while True:
                        try:
                            trial += 1
                            if trial == 2:
                                failed_example = True
                                break

                            try:
                                rec, _ = truncate_ca_qtt((accs, q_shape), rank, device='cpu')
                            except:
                                import traceback
                                traceback.print_exc()
                                continue
                            
                            detached_rec = tn.Tensor([core.detach() for core in rec.to('cpu').cores], batch=True)

                            _, stack = get_train_features(merge_cores_to_batch(qtts + [detached_rec]), rank)
                            features_train = get_test_features(rec.to('cpu'), stack)
                            preds = regressor(features_train.cuda(), other_features.cuda())

                            qtts.append(detached_rec)
                            break
                        except:
                            import traceback
                            traceback.print_exc()
                            print([tn.norm(tt) for tt in detached_rec])
                            print('Failed qtt, trial {}'.format(trial))
                            failed_states.append((tensor, enc.cpu().state_dict()))

                            enc.cuda()
                            continue
                    if failed_example:
                        continue

                loss_batch = quantile_loss(preds, labels.cuda(), quantiles)
                metric_batch = metric(preds * init_values.reshape(-1, 1).cuda(), labels.cuda() * init_values.cuda())

                try:
                    loss_batch.backward()
                    optimizer.step()
                except:
                    import traceback
                    traceback.print_exc()

                    del qtts[-1]
                    continue

                counter += 1
                loss += loss_batch.detach().cpu().numpy()
                m += metric_batch.detach().cpu().numpy().mean()

            t1 = time.time()

            if features == 'TTE2E':
                _, stack = get_train_features(merge_cores_to_batch(qtts), rank)

                with open('{}_fold{}_training{}.pkl'.format(experiment_name, fold, j), 'wb') as f:
                    pickle.dump([encoders[-1].state_dict(), models[-1].state_dict(), stack], f)

            if (j + 1) % eval_every == 0:
                print('EVAL')
                regressor.eval()
                if features == 'TTE2E':
                    encoders[-1].eval()

                for batch_index, (tensor, other_features, labels, init_values) in enumerate(dataloaders['val']):
                    print('Batch {}'.format(batch_index))

                    if features == 'CNN':
                        resnet.train()
                        cnn_features = resnet(tensor[None].cuda())
                        preds = regressor(cnn_features, other_features.cuda())
                    elif features == 'TT':
                        preds = regressor(tensor.cuda(), other_features.cuda())
                    elif features == 'TTE2E':
                        accs = [None] * len(tensor)
                        for i, im in enumerate(tensor):
                            accs[i] = encoder(im=im)

                        trial = 0
                        failed_example = False

                        while True:
                            try:
                                trial += 1
                                if trial == 2:
                                    failed_example = True
                                    break

                                try:
                                    rec, _ = truncate_ca_qtt((accs, q_shape), rank, device='cpu')
                                except:
                                    import traceback
                                    traceback.print_exc()
                                    continue

                                features_val = get_test_features(rec.to('cpu'), stack)
                                preds = regressor(features_val.cuda(), other_features.cuda())
                                break
                            except:
                                import traceback
                                traceback.print_exc()
                                print([tn.norm(tt) for tt in detached_rec])
                                print('Failed qtt, trial {}'.format(trial))
                        if failed_example:
                            continue

                    loss_batch = quantile_loss(preds, labels.cuda(), quantiles)
                    metric_batch = metric(preds * init_values.reshape(-1, 1).cuda(), labels.cuda() * init_values.cuda())

                    counter_val += 1
                    loss_val += loss_batch.detach().cpu().numpy()
                    m_val += metric_batch.detach().cpu().numpy().mean()

                print('Epoch {} took {:.2f} s.:'.format(j + 1, t1 - t0))
                print('\t TRAIN LOSS={}'.format(loss / counter))
                print('\t TRAIN Metric={}'.format(m / counter))
                print('\t VAL LOSS={}'.format(loss_val / counter_val))
                print('\t VAL Metric={}'.format(m_val / counter_val))

    os.makedirs(results_folder, exist_ok=True)
    with open(os.path.join(results_folder, experiment_name), 'wb') as f:
        if features == 'CNN':
            pickle.dump([(resnets[i].state_dict(), models[i].state_dict()) for i in range(len(models))] + list(folds), f)
        elif features == 'TT':
            pickle.dump([models[i].state_dict() for i in range(len(models))] + list(folds), f)
        elif features == 'TTE2E':
            pickle.dump(
                [models[i].state_dict() for i in range(len(models))] +
                    list(folds) +
                    [encoders[i].state_dict() for i in range(len(encoders)) +
                    failed_states], f)


if __name__ == '__main__':
    main()
