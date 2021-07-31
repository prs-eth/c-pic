import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import torch
import tntorch as tn
import numpy as np
import numba as nb
from utils.shape import q_coords2full
from utils.tt_approx import idxmapper


def test_q_coords2full():
    res = 1024
    low_dim = np.array([464, 832, 0])
    l = 2 * int(np.log(res) / np.log(2))
    high_dim = np.array(
        np.unravel_index(
            np.ravel_multi_index(
                low_dim, (res, res, 1), order='F'), [2]*l, order='F')).reshape(1, -1)
    
    assert np.allclose(
        q_coords2full(high_dim, np.array([2]*l), np.array([res, res, 1])),
        low_dim)
    
    assert np.allclose(
        q_coords2full(low_dim.reshape(1, -1), np.array([res, res, 1]), np.array([2]*l)),
        high_dim)



def test_idxmapper():
    shape = [2] * 10
    rec = tn.Tensor(torch.rand([1] + shape), ranks_tt=10)
    patches = idxmapper(rec[0], np.array([[0, 0, 0], [2, 2, 2]]), shape, 5)
    
    print(patches)#.shape == torch.Size([2, 5, 5, 5])

