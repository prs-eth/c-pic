import numpy as np
import torch
import tntorch as tn
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import utils.diffcross.diffcross as diffcross
from utils.shape import is_power2, fortran_reshape
from utils.shape import z_reorder, inv_z_reorder, reshape_power, inv_reshape_power


def truncate_qtt(x, r, device='cuda'):
    '''
        TTSVD for z reshape
    '''
    res = tn.Tensor(z_reorder(x), ranks_tt=r, requires_grad=True, device=device, batch=True)
    return inv_z_reorder(res.torch(), len(x.shape) - 1), res


def truncate_ca_qtt_diff(x, r, device='cuda'):
    accs, infos, q_shape = x
    res = [None] * len(accs)
    domain = [torch.arange(sh).to(torch.device(device)) for sh in q_shape] 

    for i, acc in enumerate(accs):
        function = lambda q_coords: acc(q_coords).to(torch.device(device))
        res[i], _ = diffcross.cross_forward(
            infos[i],
            function=function,
            domain=domain,
            function_arg='matrix',
            return_info=True)

    # Note(aelphy): build batch tt tensor
    cores = [core[None, ...] for core in res[0].cores]
    for i in range(1, len(res)):
        for j in range(len(res[0].cores)):
            cores[j] = torch.cat([cores[j], res[i].cores[j][None, ...]])

    return tn.Tensor(cores, batch=True), infos


# TODO(aelphy): fix batching for cross
def truncate_ca_qtt(x, r, device='cuda'):
    '''
        CA for the input tensor which is seen through accessor, shape of the tensor should be provided along with the accessor
    '''
    accs, q_shape = x
    infos = [None] * len(accs)

    domain = [torch.arange(sh).to(torch.device(device)) for sh in q_shape] 
    for i, acc in enumerate(accs):
        function = lambda q_coords: acc(q_coords).to(torch.device(device))

        _, info = tn.cross(
            function=function, domain=domain,
            function_arg='matrix', ranks_tt=r,
            return_info=True, device=device, verbose=False, suppress_warnings=True,
            detach_evaluations=True,
            max_iter=15 # Note(aelphy): not to wait very long
        )

        infos[i] = info

    return truncate_ca_qtt_diff((accs, infos, q_shape), r, device=device)


# TODO(aelphy): fix batching for cross
def truncate_ca_qtt_rand(x, r, device='cuda'):
    '''
        CA for the input tensor which is seen through accessor, shape of the tensor should be provided along with the accessor
    '''
    accs, q_shape = x
    infos = [None] * len(accs)

    domain = [torch.arange(sh).to(torch.device(device)) for sh in q_shape] 
    for i, acc in enumerate(accs):
        function = lambda q_coords: acc(q_coords).to(torch.device(device))

        _, info = tn.cross(
            function=function, domain=domain,
            function_arg='matrix', ranks_tt=r,
            return_info=True, device=device, verbose=False, suppress_warnings=True,
            detach_evaluations=True,
            max_iter=15 # Note(aelphy): not to wait very long
        )

        infos[i] = info

    return truncate_ca_qtt_diff((accs, infos, q_shape), r, device=device)


def truncate_qtt2(x, r, device='cuda'):
    '''
        z reshape, then another reshape to make final shape 4^p, then TTSVD
    '''
    b, m, n = x.shape
    assert is_power2(m) and is_power2(n) and m==n
    p = int(np.log(m) / np.log(2))

    res = tn.Tensor(
        fortran_reshape(z_reorder(x), (-1, *[4] * p)),
        ranks_tt=r, requires_grad=True, device=device, batch=True
    )

    return inv_z_reorder(fortran_reshape(res.torch(), [-1] + [2] * (2 * p))), res


def truncate_cp(x, r, device='cuda'):
    '''
        fortran reshape, then CP decomposition
    '''
    b, m, n = x.shape
    assert is_power2(m) and is_power2(n) and m==n
    p = int(np.log(m) / np.log(2))

    res = tn.Tensor(reshape_power(x), ranks_cp=r, requires_grad=True, device=device, batch=True)

    return inv_reshape_power(res.torch()), res


def truncate_tt_block(x, r, device='cuda'):
    '''
        fortran reshape, then block TTSVD (including batch dimension)
    '''
    b, m, n = x.shape
    assert is_power2(m) and is_power2(n) and m==n
    p = int(np.log(m) / np.log(2))

    res = tn.Tensor(reshape_power(x), ranks_tt=r, requires_grad=True, device=device)

    return inv_reshape_power(res.torch()), res


def truncate_tt(x, r, device='cuda'):
    '''
        fortran reshape, then TTSVD
    '''
    shape = x.shape
    if len(shape) == 3:
        b, m, n = shape
        assert is_power2(m) and is_power2(n)
    elif len(shape) == 4:
        b, m, n, k = shape
        assert is_power2(m) and is_power2(n) and is_power2(k)
    else:
        raise RuntimeError('Wrong dimantionality')

    res = tn.Tensor(reshape_power(x), ranks_tt=r, requires_grad=True, device=device, batch=True)

    return inv_reshape_power(res.torch(), shape=shape), res


def truncate_tt_eig(x, r, device='cuda'):
    '''
        fortran reshape, then TTSVD with eig algorithm
    '''
    res = tn.Tensor(reshape_power(x), ranks_tt=r, requires_grad=True, device=device, batch=True, algorithm='eig')
    return inv_reshape_power(res.torch()), res


def truncate_im_tt(x, r, device='cuda'):
    '''
        TTSVD for the original tensor
    '''
    res = tn.Tensor(x, ranks_tt=r, requires_grad=True, device=device, batch=True)
    return res.torch(), res


# Note(aelphy): do nothing, why not ? :)
def truncate_none(x, r, device='cuda'):
    return x.to(device), None


def truncate_svd(x, r):
    '''
        TTSVD for the original matrices
    '''
    dims = x.shape[-2:]
    u, s, v = torch.svd(x.reshape(-1, *dims))
    return u[:, :, :r].matmul(torch.diag_embed(s[:, :r])).matmul(torch.transpose(v[:, :, :r], 1, 2)), (u, s, v)
