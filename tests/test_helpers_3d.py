import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import torch
import numpy as np
from utils.helpers_3d import build_patch_coords, get_3dtensors_patches


def test_build_patch_coords():
    test_tensor = np.random.random((100, 100, 100))
    w = 3
    test_centers = np.random.randint(w // 2 + 1, 100, 10 * 3).reshape(10, 3)

    comparison = get_3dtensors_patches(
        torch.tensor(test_tensor)[None], w, np.hstack([np.zeros((10, 1), dtype=np.int64), test_centers]))

    coords = build_patch_coords(test_centers, w)

    assert np.allclose(
        test_tensor[coords[:, 0], coords[:, 1], coords[:, 2]].reshape(10, 3, 3, 3), comparison)
