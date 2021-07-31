import os
import os.path as osp
import numpy as np

def iterate_minibatches(l, batch_size, shuffle=True, full=True):
    idxs = np.arange(l)
    if shuffle:
        np.random.shuffle(idxs)

    if batch_size > l:
        yield idxs
        return
    
    if full:
        for start_idx in range(0, l, batch_size):
            excerpt = idxs[start_idx:start_idx + batch_size]
            yield excerpt
    else:
        excerpt = idxs[0:batch_size]
        yield excerpt

def get_latest_wights_fname(weights_path, name):
    weight_fnames = os.listdir(weights_path)
    if len(weight_fnames) == 0:
        i = 0
    else:
        m = 0
        for weight_fname in list(reversed(sorted(weight_fnames))):
            if weight_fname.startswith(name):
                i = int(weight.split('_')[-1].split('.')[0])
                if i > m:
                    m = i
        i = m

    return osp.join(weights_path, name + '{}.pkl'.format(i)), i
