#!/cluster/apps/python/3.6.4_gpu/bin/python

import argparse
import os
import os.path as osp

import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

parser = argparse.ArgumentParser(description='TT learning experiments on mnist')
parser.add_argument('--rank', type=int, help='rank')
parser.add_argument('--problem', type=str, help='classification or regression')
parser.add_argument('--data', type=str, help='data for experiment: on single image, one class, or full')
parser.add_argument('--digit', type=int, help='digit for single and one class')
parser.add_argument('--truncation', type=str, help='truncate function')
args = parser.parse_args()

from mnist import MNIST

import numpy as np
import cv2
import torch
import tntorch as tn

from sklearn.model_selection import train_test_split

from utils.svd import truncate_svd, truncate_qtt, truncate_qtt2, truncate_tt
from utils.train import train_mse, train_tt_ce, compute_MSE, compute_CE
from utils.stats import save_stats
from utils.radam import RAdam
from utils.tt_approx import compute_batch_cores, compute_batch_cores_svd
from models.cnn import Encoder, Decoder, TTClassifier

NUM_CLASSES = 10

mndata = MNIST('../data/mnist/')
images, labels = mndata.load_training()
images = np.array(images)
labels = np.array(labels)

torch.set_anomaly_enabled(True)
torch.set_default_dtype(torch.float64)

images_big = []

for img in images:
    images_big.append(cv2.resize(img.reshape(28, 28).astype(np.uint8), (32, 32)) / 255.)

images_big = np.array(images_big)

np.random.seed(42)
dims = (32, 32)
sample = torch.tensor(images_big[labels == 0][0].reshape(1, *dims))

r = args.rank
print('rank {}'.format(r))

rec, t = globals()[args.truncation](torch.cat([sample, sample]), r)

if args.truncation == 'truncate_svd':
    batch_cores_function = compute_batch_cores_svd
else:
    batch_cores_function = compute_batch_cores

model1 = Encoder().cuda()
model2 = Decoder().cuda()
model3 = TTClassifier(batch_cores_function(t), NUM_CLASSES).cuda()
optimizer = RAdam(list(model1.parameters()) + list(model2.parameters()) + list(model3.parameters()), lr=1e-3)

if args.problem == 'classification' and args.data != 'all':
    raise RuntimeError('canot learn classification not on all data')

if args.problem == 'regression':
    if args.data == 'single':        
        idxs = labels == args.digit
        idx = np.random.choice(idxs.sum())
        
        print('training on single instace of {} number {}'.format(args.digit, idx))
        sample = torch.tensor(images_big[idxs][idx].reshape(1, *dims))
        train_mse(
            sample, sample,
            r, 1000, 1, 100,
            optimizer, globals()[args.truncation], model1, model2,
            str(args.digit), save_stats, torch.tensor([args.digit]), torch.tensor([args.digit])
        )
    elif args.data == 'class':
        X_train, X_val, y_train, y_val = train_test_split(images_big[labels==args.digit], labels[labels==args.digit])
        train_mse(
            X_train, X_val,
            r, 1000, 320, 10,
            optimizer, globals()[args.truncation], model1, model2,
            'all_{}'.format(args.digit), save_stats, torch.tensor(y_train), torch.tensor(y_val)
        )
    elif args.data == 'all':
        print ("training on all images") 
        X_train, X_val, y_train, y_val = train_test_split(images_big, labels)
        train_mse(
            X_train, X_val,
            r, 1000, 320, 10,
            optimizer, globals()[args.truncation], model1, model2,
            'all', save_stats
        )
elif args.problem == 'classification':
    X_train, X_val, y_train, y_val = train_test_split(images_big, labels)
    train_tt_ce(
        X_train, X_val, torch.tensor(y_train), torch.tensor(y_val),
        r, 1000, 320, 10, optimizer, globals()[args.truncation],
        model1, model2, model3,
        save_stats, batch_cores_function
    )
