import os
import os.path as osp
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser(description='merge results on folder')
parser.add_argument('--reference_path', type=str, help='reference_path')
parser.add_argument('--target_path', type=str, help='target_path')
args = parser.parse_args()

paths1 = []
for fname in os.listdir(args.reference_path):
    paths1.append(fname)
    

paths2 = []

for fname in os.listdir(args.target_path):
    paths2.append('_'.join(fname.split('_')[:2]))
    
for fname in set(paths1) - set(paths2):
    print(fname)
