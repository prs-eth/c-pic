import numpy as np
import torch.nn.functional as F
import torch
from typing import Union

# Note(aelphy): may be optimized
# TODO(aelphy): make it work with color images

# Cuts image patches from image at certain locations
# im: grayscale image, np.array or torch.Tensor b \times N \times M
# rec_field: receptieve field required, int
# coords: centers of patches, np.array e.g., np.array([[b_1, x_1, y_1], [b_2, x_2, y_2]])
# Returns: torch.Tensor of size b \times rec_field \times rec_field
def get_imgs_patches(
    imgs: Union[np.array, torch.Tensor],
    rec_field: int,
    coords: np.array):
    res = [None] * coords.shape[0]

    assert rec_field % 2 != 0
    w = (rec_field - 1) // 2

    for i in range(len(res)):
        patch = imgs[
            coords[i][0],
            max(coords[i][1] - w, 0) : coords[i][1] + w + 1,
            max(coords[i][2] - w, 0) : coords[i][2] + w + 1
        ]
        res[i] = F.pad(
            patch,
            [- min(coords[i][2] - w, 0),
             - min(imgs.shape[2] - coords[i][2] - w - 1, 0),
             - min(coords[i][1] - w, 0),
             - min(imgs.shape[1] - coords[i][1] - w - 1, 0)
            ], # left, right, top, bottom,
            mode='constant', value=0
        )[None, ...]

    return torch.cat(res)
