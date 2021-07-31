from typing import Iterable

import torch
import itertools
import tntorch as tn
import numpy as np
from typing import Sequence


def merge_cores_to_batch(tts: Sequence[tn.Tensor]) -> tn.Tensor:
    '''
    Merge batched cores into one large batched core
    '''
    cores = []
    for i in range(len(tts[0].cores)):
        cores.append(torch.cat([tts[j].cores[i] for j in range(len(tts))]))
    return tn.Tensor(cores, batch=True)


def get_train_features(batch, nfeatures):
    """
    :param batch: a batched tensor representing B QTTs
    :param nfeatures: number of desired features to extract
    :return: features (B x nfeatures matrix) and stack (the orthogonalized stack)
    """
    
    B = batch.shape[0]
    
    def insert_dim(batch):
        for b in range(B):
            t = tn.Tensor([c[b, ...] for c in batch.cores])  # Get b-th instance from the batch
            yield tn.unsqueeze(t, dim=0)

    stack = tn.reduce(insert_dim(batch), tn.cat, dim=0, algorithm='eig', rmax=nfeatures)
    return stack.cores[0][0, :, :], stack

 
def get_test_features(batch, stack):
    """
    :param batch: batched QTT containing K test instances
    :param stack: as given by `get_train_features()`
    :return: matrix of shape K x nfeatures
    """

    N = stack.dim()-1
    B = batch.shape[0]
    
    c = torch.ones(B, 1, 1)
    for n in range(N-1, -1, -1):
        c = torch.einsum('jai,bik->bjak', (stack.cores[n + 1], c))  # Shape: B x Rl x I x Sr
        c = torch.einsum('biak,bjak->bji', (batch.cores[n], c))     # Shape: B x Rl x Sl
    c = c[..., 0]
    return c


def idxmapper(t: tn.Tensor, idxs: np.array, shape: Iterable, K: int):
    """
    Retrieves a collection of P patches, each of size K^D, from a D-dimensional QTT tensor of shape 2^(s_1 + s_2 + ... + s_D))

    Note: patch coordinates will be clamped to valid values [0, ..., 2^s_i - 1]

    :param t: a non-batch 2^(s_1 + s_2 + ... + s_D) tensor in QTT (without transpose) format
    :param idxs: a P x D matrix with the centers of all P patches
    :param K: the patch size along each dimension
    :return: a P x K x ... x K = P x K^D torch tensor
    """

    assert all([sh == 2 for sh in t.shape])
    D = idxs.shape[1]

    assert np.prod(t.shape) == np.prod(shape)
    j = 0

    for i in range(D):
        # Compute patch coordinates along this dimension
        column = idxs[:, i, None] + np.arange(-(K - 1) // 2, (K - 1) // 2 + 1)[None, :]
        N = int(np.log(shape[i]) / np.log(2))

        # Clamp to valid indices [0, ..., 2^N-1]
        column[column < 0] = 0
        column[column > 2 ** N - 1] = 2 ** N - 1

        # Convert to binary indices
        column = np.array(np.unravel_index(column.flatten(), [2] * N, order='F'))

        # Access the QTT tensor at this dimension
        t = t[[slice(None)] * j + list(column)]
        
        if j < D - 1:
            j += 1

    # Convert result to a batched tntorch tensor
    for n in range(t.dim()):
        t.cores[n] = torch.reshape(t.cores[n], [t.cores[n].shape[0], idxs.shape[0], -1, t.cores[n].shape[-1]])
        t.cores[n] = t.cores[n].permute(1, 0, 2, 3)
    t.batch = True
    return t.torch()


def compute_batch_cores(t, device='cuda'):
    return [core.reshape(core.shape[0], -1).to(device) for core in t.cores]


def compute_batch_cores_svd(t, device='cuda'):
    return [i.reshape(i.shape[0], -1).to(device) for i in t]


def build_square_domain(res, dim=3):
    ds = [res] * dim
    return [torch.arange(ds[n]) for n in range(dim)]


def build_linspace_domain(mins, maxs, res):
    return [torch.linspace(mins[i], maxs[i], res) for i in range(maxs.shape[0])]


def metrics(full, t):
    print(t)
    print('Compression ratio: {}/{} = {:g}'.format(full.numel(), t.numcoef(), full.numel() / t.numcoef()))
    print('Relative error:', tn.relative_error(full, t))
    print('RMSE:', tn.rmse(full, t))
    print('R^2:', tn.r_squared(full, t))
    

def save_results(model_name, t, d, res):
    t_np = t.numpy()
    nonzero_idxs = t_np.nonzero()
    rec = np.concatenate([nonzero_idxs, [t_np[nonzero_idxs]]]).T
    orig = np.vstack(d.nonzero()).T
    np.savetxt('{}_rec_{}.xyz'.format(model_name, res), rec)
    np.savetxt('{}_orig_{}.xyz'.format(model_name, res), orig)


def reduce(domain, initializers, expressions, finalizers, cut=128**3):
    assert len(initializers) == len(expressions) == len(finalizers)
    storages = []
    
    for i in range(len(initializers)):
        storages.append({})
        initializers[i](storages[-1])
        
    i = 0
    buffer = [None] * cut

    for index_set in itertools.product(*[np.arange(len(d)) for d in domain]):
        if i < cut:
            buffer[i] = index_set
            i += 1
        else:
            i = 0
            for j in range(len(expressions)):
                expressions[j](np.array(buffer), domain, storages[j])
            buffer = [None] * cut

    if i > 0:
        buffer = buffer[:i]
        for j in range(len(expressions)):
                expressions[j](np.array(buffer), domain, storages[j])

    return [finalizers[i](storages[i]) for i in range(len(storages))]
