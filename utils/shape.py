import numpy as np
import torch
from numba import njit

def is_power2(num):
    'checks if a number is a power of two'
    return num != 0 and ((num & (num - 1)) == 0)
    
    
def fortran_reshape(x, shape, batch=True):
    if batch:
        return x.permute([0] + list(np.arange(1, len(x.shape))[::-1]))\
                .reshape([shape[0]] + list(reversed(shape[1:])))\
                .permute([0] + list(np.arange(1, len(shape))[::-1])).contiguous()
    else:
        return x.permute(*np.arange(len(x.shape))[::-1])\
                .reshape(list(reversed(shape)))\
                .permute(*np.arange(len(shape))[::-1]).contiguous()


# Batched z_reorder
def z_reorder(A):
    shape = A.shape
    nmb_dim = len(shape)
    
    assert all([is_power2(shape[i]) for i in range(1, nmb_dim)])
    assert all([shape[i] == shape[i + 1] for i in range(1, nmb_dim - 1)])

    p = sum([int(np.log(shape[i + 1]) / np.log(2)) for i in range(nmb_dim - 1)])
    dims = list(np.ravel([(a, b) for (a, b) in zip(np.arange(1, p // 2 + 1), np.arange(1, p // 2 + 1) + p // 2)]))
    if p % 2 != 0:
        dims += [p]
        
    return reshape_power(A).permute(0, *dims)


# Batched inv_z_reorder
def inv_z_reorder(A, nmb_dim=2):
    shape = A.shape
    dim = np.prod(shape[1:])
    assert is_power2(dim)
    p = int(np.log(dim) / np.log(2) / nmb_dim)
    dims = list(np.concatenate([np.arange(1, nmb_dim * p + 1)[::2], np.arange(1, nmb_dim * p + 1)[1::2]]))        
    tmp_cstyle = A.permute(0, *dims)
    return fortran_reshape(tmp_cstyle, [shape[0], *[2**p for _ in range(nmb_dim)]])


def reshape_power(A):
    '''
        Works for tensors of shape (batch_size, a^n)
    '''
    shape = A.shape
    nmb_dim = len(shape)
    
    assert all([is_power2(shape[i]) for i in range(1, nmb_dim)])
    
    p = int(np.log(np.prod(shape[1:])) / np.log(2)) 
    return fortran_reshape(A, [shape[0], *[2] * p])


def inv_reshape_power(A, nmb_dim=2, shape=None):
    '''
        reverts QTT tensor back to the original shape, if shape is not given, it will assume tensor with equal modes
    '''
    ar = np.prod(A.shape[1:])
    if shape is None:
        return fortran_reshape(A, [-1] + [int(np.ceil(ar**(1 / nmb_dim))) for _ in range(nmb_dim)])
    else:
        return fortran_reshape(A, shape)


def full_coords2z(x, limits, target_shape):
    assert np.prod(limits) == np.prod(target_shape)
    
    d = x.shape[-1]
    p = len(target_shape) // 2
    
    assert p * 2 == len(target_shape)

    device = x.device
    
    z_permute_idxs = np.ravel([(a, b) for (a, b) in zip(np.arange(p), np.arange(p) + p)])
    pows = torch.tensor([np.prod(limits[d-i:]) for i in range(d)], device=device).long()
    
    return np.vstack(
        np.unravel_index(
            x.matmul(pows).long().detach().cpu().numpy(),
            target_shape,
            order='F'
        )
    ).T[:, z_permute_idxs]


def z_coords2full(x, limits, target_shape):
    assert np.prod(limits) == np.prod(target_shape)
    
    d = x.shape[-1]
    p = d // 2
    
    assert p * 2 == d
    
    device = x.device
    
    z_permute_idxs = np.ravel([(a, b) for (a, b) in zip(np.arange(p), np.arange(p) + p)])    
    pows = torch.tensor([np.prod(limits[d-i:]) for i in range(d)], device=device)
    
    return np.unravel_index(x.matmul(pows[z_permute_idxs]).long().detach().cpu().numpy(), target_shape, order='F')


@njit
def q_coords2full(x: np.array, source_shape: np.array, target_shape: np.array):
    """
    Translate indices between two tensor shapes (both must consist of powers of two), in Fortran ordering
    :param x: a matrix (one row per index)
    :param source_shape: a vector of powers of 2
    :param target_shape: another vector of powers of 2
    :return: the translated matrix (one row per index)
    """

    assert np.sum(np.log2(source_shape)) == np.sum(np.log2(target_shape))

    result = np.zeros((len(x), len(target_shape)), dtype=np.int64)
    for i in range(len(x)):
        for j in range(len(source_shape)):
            assert x[i, j] < source_shape[j]
        
        source_pos = 0
        source_bitpos = 0
        for target_pos in range(len(target_shape)):
            result[i, target_pos] = 0
            target_bitpos = 0
            for _ in range(int(np.log2(target_shape[target_pos]))):
                result[i, target_pos] += ((x[i, source_pos] >> source_bitpos) & 1) << target_bitpos
                target_bitpos += 1
                source_bitpos += 1
                if source_bitpos == int(np.log2(source_shape[source_pos])):
                    source_bitpos = 0
                    source_pos += 1
    return result
