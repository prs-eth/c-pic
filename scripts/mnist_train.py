# for res in 16 32 64 128
# do
#     for rank in 3 5 7 10 15 20
#     do
#         bsub -R "rusage[mem=15000,ngpus_excl_p=1]" -W 120:00 python mnist_train.py --batch_size=2 --rank=$rank --num_epochs=1 --eval_every=3 --res=$res --truncation=truncate_ca_qtt --dataset=ModelNet --num_classes=10 --data_path=../data --dense=True --build_enc_f=build_enc_acc3d --patch_function=get_3dtensors_patches --df=False
#     done
# done

# for res in 16 32 64 128 256
# do
#     for rank in 3 5 7 10 15 20
#     do
#         bsub -R "rusage[mem=15000,ngpus_excl_p=1]" -W 120:00 python mnist_train.py --batch_size=2 --rank=$rank --num_epochs=1 --eval_every=3 --res=$res --truncation=truncate_ca_qtt --dataset=ModelNet --num_classes=10 --data_path=../data --dense=True --build_enc_f=build_df_enc_acc3d --patch_function=get_df_patches --df=True
#     done
# done

import sys
sys.path.append('../')

import numpy as np
import time
import os
import os.path as osp
import cv2
import open3d as o3d
import torch
import pickle
import importlib
import argparse
from sklearn.model_selection import train_test_split

import utils
from utils.svd import truncate_ca_qtt, truncate_tt, truncate_ca_qtt_diff
from utils.train import train_mse_dl, train_ca_mse_dl, eval_ca_mse_dl, compute_CE
from utils.shape import inv_reshape_power, reshape_power, q_coords2full
from utils.accessors import build_enc_acc2d, build_no_enc_acc2d, get_imgs_patches, get_df_patches
from utils.accessors import build_enc_acc3d, build_no_enc_acc3d, get_3dtensors_patches, build_df_enc_acc3d
from utils.radam import RAdam
from utils.misc import get_latest_wights_fname
from models.cnn import EncoderCA, DecoderCA
import utils.dataloaders as dataloaders

torch.set_default_dtype(torch.float64)


def extract_latent_tensor(dl, encoder, q_shape, r, res, label_encoder, df):
    X = []
    y = []
    for batch in dl:
        encs = []
        if df:
            for x in batch[0]:
                encs.append(build_df_enc_acc3d(x, encoder, q_shape, x.shape, True, 10000))
        else:
            for x in batch['data'][str(res)]:
                encs.append(build_enc_acc3d(x.double(), encoder, q_shape, x.shape, True, 10000))

        rec, _ = truncate_ca_qtt([encs, q_shape], r)
        rec.to('cpu').as_leaf()

        X.append(rec)
        if df:
            y.append(batch[1])
        else:
            y.append(label_encoder.transform(batch['attributes']['category']))
        
    cores = [None] * len(X[0].cores)
    for i in range(len(cores)):
        cores[i] = torch.cat([X[j].cores[i] for j in range(len(X))])
    
    return tn.Tensor(cores, batch=True), torch.concatenate(y)


parser = argparse.ArgumentParser(description='Reconstruction MSE')
parser.add_argument('--batch_size', type=int, required=True, help='batch size')
parser.add_argument('--num_epochs', type=int, required=True, help='num epochs')
parser.add_argument('--eval_every', type=int, required=True, help='how often to perform evaluation')
parser.add_argument('--rank', type=int, required=True, help='rank')
parser.add_argument('--res', type=int, required=True, help='resolution')
parser.add_argument('--truncation', type=str, required=True, help='truncation function: truncate_ca_qtt or truncate_tt')
parser.add_argument('--dataset', type=str, required=True, help='dataset')
parser.add_argument('--num_classes', type=int, required=True, help='number of classes')
parser.add_argument('--data_path', type=str, required=True, help='dataset base path')
parser.add_argument('--dense', type=bool, required=True, help='use dense models ?')
parser.add_argument('--build_enc_f', type=str, required=True, help='function to build encoder')
parser.add_argument('--patch_function', type=str, required=True, help='function to extract patches')
parser.add_argument('--df', type=str, required=True, help='df')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
EVAL_EVERY = args.eval_every
NUM_EPOCHS = args.num_epochs

r = args.rank
dataset = args.dataset
num_classes = args.num_classes
res = args.res

assert args.df in ['True', 'False']
df = args.df == 'True'

MODELNET_ROOT = osp.join(args.data_path, dataset + str(num_classes))
if df:
    get_dataset = getattr(dataloaders, 'get_{}_DF_dataloaders'.format(dataset))
else:
    get_dataset = getattr(dataloaders, 'get_{}_dataloaders'.format(dataset))

dl_train, dl_val, label_encoder = get_dataset(res, MODELNET_ROOT, BATCH_SIZE)
this_module = sys.modules[__name__]

print('rank {}'.format(r))
truncation = args.truncation
print('truncation {}'.format(truncation))
print('resolution {}'.format(res))
print('df {}'.format(df))

assert truncation in ['truncate_ca_qtt', 'truncate_tt']

if truncation == 'truncate_ca_qtt':
    name = 'qtt_ca_df{}_{}_rank{}_res{}'.format(df, dataset, r, res)
    q_shape = [2] * int(np.log(res**3) / np.log(2))

    if args.dense:
        from models.dense import Encoder, Decoder, Classifier

        encoder = Encoder([5**3, 128, 256, 1], 3).cuda()
        decoder = Decoder([5**3, 128, 256, 1], 3).cuda()
    else:
        encoder = EncoderCA().cuda()
        decoder = DecoderCA().cuda()
else: #  truncation == 'truncate_tt'
    name = 'qtt_ttsvd_df{}_{}_rank{}_res{}'.format(df, dataset, r, res)
    
    from models.cnn import Encoder, Decoder
    encoder = Encoder().cuda()
    decoder = Decoder().cuda()
    
optimizer = RAdam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

weights_path = '../weights'
results_path = '../results'

weight_fpath, i = get_latest_wights_fname(weights_path, name)

if osp.exists(weight_fpath):
    with open(weight_fpath, 'rb') as f:
        encoder_state_dict, decoder_state_dict, optimizer_state_dict = pickle.load(f)

    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    i += 1
    weight_fpath = osp.join(weights_path, name + '{}_res{}.pkl'.format(i, res))

for j in range(NUM_EPOCHS):
    if truncation == 'truncate_ca_qtt':
        train_ca_mse_dl(
            dl_train,
            dl_val, 
            r=r, res=res, num_epochs=1,
            eval_every=EVAL_EVERY,
            optimizer=optimizer, trunc=truncate_ca_qtt, 
            trunc_diff=truncate_ca_qtt_diff,
            encoder=encoder, decoder=decoder, 
            build_enc_f=getattr(utils.accessors, args.build_enc_f), 
            patch_function=getattr(utils.accessors, args.patch_function),
            coords2full=q_coords2full, inv_reorder=inv_reshape_power, 
            q_shape=q_shape,
            micro_batch_size=3000,
            weights_file_name=weight_fpath
        )
    elif truncation == 'truncate_tt':
        train_mse_dl(
            dl_train,
            dl_val,
            r, 1, BATCH_SIZE, 2, optimizer, truncate_tt, encoder, decoder
        )

if truncation == 'truncate_ca_qtt':
    mse_val = eval_ca_mse_dl(
        dl_val,
        r=r, res=res,
        trunc=truncate_ca_qtt,
        trunc_diff=truncate_ca_qtt_diff,
        encoder=encoder, decoder=decoder,
        build_enc_f=getattr(utils.accessors, args.build_enc_f),
        patch_function=getattr(utils.accessors, args.patch_function),
        coords2full=q_coords2full, inv_reorder=inv_reshape_power,
        q_shape=q_shape,
        micro_batch_size=3000
    )

    latent_tensor_train, y_train = extract_latent_tensor(dl_train, encoder, q_shape, r, res, label_encoder, df)
    latent_tensor_val, y_val = extract_latent_tensor(dl_val, encoder, q_shape, r, res, label_encoder, df)

    from utils.tt_approx import get_train_features, get_test_features

    clf_mlp = Classifier(r, len(np.unique(y_train)))
    optimizer_mlp = RAdam(
        list(clf_mlp.parameters()),
        lr=1e-3, weight_decay=1e-3
    )

    idxs = torch.argsort(y_train)
    features_train, stack = get_train_features(latent_tensor_train[idxs], r)
    features_val = get_test_features(latent_tensor_val, stack)
    features_train_val = get_test_features(latent_tensor_train[idxs], stack)

    best_accuracy = -np.inf

    for k in range(1000):
        clf_mlp.train()
        optimizer_mlp.zero_grad()
        logits = clf_mlp(features_train_val.detach())
        loss = compute_CE(y_train[idxs], logits)

        loss.backward()
        optimizer_mlp.step()

        if (k + 1) % 10 == 0 or k == 0:
            clf_mlp.eval()
            logits_val = clf_mlp(features_val.detach())
            loss_val = compute_CE(y_val, logits_val)
            preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
            preds_val = torch.argmax(torch.softmax(logits_val, dim=-1), dim=-1)
            accuracy_val = (preds_val == y_val).sum() / float(len(y_val))
            accuracy = (preds == y_train[idxs]).sum() / float(len(y_train))

            if accuracy_val > best_accuracy:
                best_accuracy = accuracy_val

    with open(osp.join(results_path, name + '_{}.pkl'.format(i)), 'wb') as f:
        pickle.dump(
            [
                clf_mlp.state_dict(), optimizer_mlp.state_dict(),
                mse_val, best_accuracy
            ],
            f
        )
