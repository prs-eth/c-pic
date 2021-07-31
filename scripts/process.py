#!/cluster/apps/python/3.6.4_cpu/bin/python

import argparse
import os
import os.path as osp
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.helpers_3d import read_pcl, compute_max_possible_distance
from utils.tt_approx import reduce, build_square_domain, build_linspace_domain, build_distance_metric
from utils.grid_functions import build_r2, build_RMSE, build_relative_error
from utils.sparse_tools import sparse_to_dense
import tntorch as tn
import torch
import numpy as np
import itertools
import pickle

parser = argparse.ArgumentParser(description='TT experiments on point clouds')
parser.add_argument('--scannet_path', type=str, help='scannet_path')
parser.add_argument('--results_path', type=str, help='results_path')
parser.add_argument('--scene', type=str, help='scene')
parser.add_argument('--res', type=int, help='res')
args = parser.parse_args()

if osp.exists(osp.join('data', 'mins_maxs_scannet.pkl')):
    with open(osp.join('data', 'mins_maxs_scannet.pkl'), 'rb') as f:
        mins, maxs = pickle.load(f)
else:        
    mins1 = np.array([np.inf, np.inf, np.inf])
    mins2 = np.array([np.inf, np.inf, np.inf])
    maxs1 = np.array([-np.inf, -np.inf, -np.inf])
    maxs2 = np.array([-np.inf, -np.inf, -np.inf])

    mins = [mins1, mins2]
    maxs = [maxs1, maxs2]

    for scene in os.listdir(args.scannet_path):
        v1_path = osp.join(args.scannet_path, args.scene, args.scene + '_vh_clean' + '.ply')
        v2_path = osp.join(args.scannet_path, args.scene, args.scene + '_vh_clean_2' + '.ply')

        for i, p in enumerate((v1_path, v2_path)):
            pcl = read_pcl(p)
            min_l = np.asarray(pcl.points).min(axis=0)
            max_l = np.asarray(pcl.points).max(axis=0)
            mins[i] = np.where(mins[i] < min_l, mins[i], min_l)
            maxs[i] = np.where(maxs[i] > max_l, maxs[i], max_l)
            
    with open(osp.join('data', 'mins_maxs_scannet.pkl'), 'wb') as f:
        pickle.dump((mins, maxs), f)
        
domain1 = build_linspace_domain(mins[0], maxs[0], args.res)
domain2 = build_linspace_domain(mins[1], maxs[1], args.res)
domains = [domain1, domain2]

results = {0: {}, 1: {}}

for i, p in enumerate((
    osp.join(args.scannet_path, args.scene, args.scene + '_vh_clean' + '.ply'),
    osp.join(args.scannet_path, args.scene, args.scene + '_vh_clean_2' + '.ply')
)):
    max_possible_distance = compute_max_possible_distance(domains[i])
    pcl = read_pcl(p)

    measure = build_distance_metric(pcl, max_possible_distance)

    approx = tn.cross(
        function=measure,
        domain=domains[i],
        function_arg='matrix',
        ranks_tt=3, val_size=20000, max_iter=1, verbose=0
    )

    with open(osp.join(args.scannet_path, args.scene, '{}_{}_{}.tt'.format(args.scene, i, args.res)), 'wb') as f:
        pickle.dump(approx.cores, f)

    results[i][args.scene] = {}
    results[i][args.scene]['compression_ratio']  = len(pcl.points) / approx.numcoef()
    results[i][args.scene]['Relative error'], results[i][args.scene]['RMSE'], results[i][args.scene]['R^2'] = reduce(
        [d.numpy() for d in domains[i]],
        *list(
            zip(
                build_relative_error(measure, approx),
                build_RMSE(measure, approx),
                build_r2(measure, approx)
            )
        )
    )
    
with open(osp.join(args.results_path, args.scene + '_{}_stats.pkl'.format(args.res)), 'wb') as f:
    pickle.dump(results, f)
