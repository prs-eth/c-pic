#!/cluster/apps/python/3.6.4_cpu/bin/python

import argparse
import os
import os.path as osp
import sys
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='merge results on folder')
parser.add_argument('--results_path', type=str, help='results_path')
args = parser.parse_args()

results_folder = osp.basename(args.results_path)
if not results_folder:
    osp.dirname(args.results_path)
    
res = results_folder.split('_')[-1]
output_fname = 'results_{}.pkl'.format(res)
result = {}


def merge(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a
    

for fname in tqdm(os.listdir(args.results_path)):
    if fname != output_fname:
        with open(osp.join(args.results_path, fname), 'rb') as f:
            merge(result, pickle.load(f))
            
with open(osp.join(args.results_path, output_fname), 'wb') as f:
    pickle.dump(result, f)
