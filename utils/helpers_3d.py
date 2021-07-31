import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import torch.nn.functional as F
import torch


def build_patch_coords(coords: np.array, patch_size: int):
    '''
        create patches aroung given point and return per point coordinate
    '''
    d = coords.shape[1]
    K = patch_size
    delta = np.arange(-(K - 1) // 2, (K - 1) // 2 + 1)
    columns = []

    for i in range(d):
        columns.append((coords[:, i, None] + delta[None, :]))
    patches = [None] * len(coords)
    for j in range(len(coords)):
        patchs_idxs = np.meshgrid(*[columns[i][j] for i in range(d)], indexing='ij')

        patches[j] = np.vstack([idx.ravel() for idx in patchs_idxs]).T

    return np.vstack(patches)


# TODO(avideret): generalize to n-dim tensor
# Note(aelphy): may be optimized
def get_3dtensors_patches(tensors, rec_field, coords):
    '''
        Cuts patches from tensors at certain locations
        tensors: 3d tensors, torch.Tensor b \times N \times M \times L
        rec_field: receptieve field required, int
        coords: centers of patches, np.array e.g., np.array([[x_1, y_1, z_1], [x_2, y_2, z_2]])
        Returns: torch.Tensor of size b \times rec_field \times rec_field \times rec_field
    '''
    res = [None] * coords.shape[0]

    assert rec_field % 2 != 0
    w = (rec_field - 1) // 2

    for i in range(len(res)):
        patch = tensors[
            coords[i][0],
            max(coords[i][1] - w, 0): coords[i][1] + w + 1,
            max(coords[i][2] - w, 0): coords[i][2] + w + 1,
            max(coords[i][3] - w, 0): coords[i][3] + w + 1
        ]
        res[i] = F.pad(
            patch,
            # left, right, top, bottom,
            [- min(coords[i][3] - w, 0),
             - min(tensors.shape[3] - coords[i][3] - w - 1, 0),
             - min(coords[i][2] - w, 0),
             - min(tensors.shape[2] - coords[i][2] - w - 1, 0),
             - min(coords[i][1] - w, 0),
             - min(tensors.shape[1] - coords[i][1] - w - 1, 0)],
            mode='constant', value=0
        )[None, ...]

    return torch.cat(res)
