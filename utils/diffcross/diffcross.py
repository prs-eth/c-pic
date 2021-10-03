import tntorch as tn
import torch
import numpy as np
import cvxpy as cp


# The derivative of lstsq() is not implemented as of PyTorch 1.9.0,
# so we use cvxpylayer solver
def lstsq(b, A, method='qr', lam=1e-6, eps=1e-3):
    if method == 'qr':
        Q, R = torch.linalg.qr(A.T)
        X = b.T @ torch.linalg.pinv(R) @ Q.T

        return [X.T]
    elif method == 'cvxpylayers':
        from cvxpylayers.torch import CvxpyLayer
        assert b.shape[0] == A.shape[0]

        m, n = A.shape
        m, k = b.shape

        Acp = cp.Parameter((m, n))
        bcp = cp.Parameter((m, k))
        x = cp.Variable((n, k))
        obj = cp.sum_squares(Acp @ x - bcp) + lam * cp.pnorm(x, p=2)**2
        prob = cp.Problem(cp.Minimize(obj))
        prob_th = CvxpyLayer(prob, [Acp, bcp], [x])

        return prob_th(A, b, solver_args={'eps': eps})
    else:
        raise ValueError('Wrong value of method parameter, only qr and cvxpylayers are supported')


def cross_forward(info, function=lambda x: x, domain=None, tensors=None, function_arg='vectors', detach_evaluations=True, return_info=False):
    """
    Given TT-cross indices and a black-box function (to be evaluated on an arbitrary grid), computes a differentiable TT tensor as given by the TT-cross interpolation formula.

    Reference: I. Oseledets, E. Tyrtyshnikov: `"TT-cross Approximation for Multidimensional Arrays" (2009) <http://www.mat.uniroma2.it/~tvmsscho/papers/Tyrtyshnikov5.pdf>`_

    :param info: dictionary with the indices returned by `tntorch.cross()`
    :param function: a function $\mathbb{R}^M \to \mathbb{R}$, as in `tntorch.cross()`
    :param domain: domain where `function` will be evaluated on, as in `tntorch.cross()`
    :param tensors: list of $M$ TT tensors where `function` will be evaluated on
    :param function_arg: type of argument accepted by `function`. See `tntorch.cross()`
    :param detach_evaluations: Boolean, if True, will remove gradient buffers for the function when not selected
    :param return_info: Boolean, if True, will also return a dictionary with informative metrics about the algorithm's outcome

    :return: a TT :class:`Tensor`(if `return_info`=True, also a dictionary)
    """

    assert domain is not None or tensors is not None
    assert function_arg in ('vectors', 'matrix')
    device = None
    if function_arg == 'matrix':
        def f(*args):
            return function(torch.cat([arg[:, None] for arg in args], dim=1))
    else:
        f = function
    if tensors is None:
        tensors = tn.meshgrid(domain)
        device = domain[0].device
    if not hasattr(tensors, '__len__'):
        tensors = [tensors]

    Is = list(tensors[0].shape)
    N = len(Is)

    # Load index information from dictionary
    lsets = info['lsets']
    rsets = info['rsets']
    left_locals = info['left_locals']
    Rs = info['Rs']

    if return_info:
        info['Xs'] = torch.zeros(0, N)
        info['shapes'] = []

    assert function_arg in ('vectors', 'matrix')
    if function_arg == 'matrix':
        def f(*args):
            return function(torch.cat([arg[:, None] for arg in args], dim=1))
    else:
        f = function

    # Initialize left and right interfaces for `tensors`
    def init_interfaces():
        t_linterfaces = []
        t_rinterfaces = []
        for t in tensors:
            linterfaces = [torch.ones(1, t.ranks_tt[0]).to(device)] + [None]*(N-1)
            rinterfaces = [None] * (N-1) + [torch.ones(t.ranks_tt[t.dim()], 1).to(device)]
            for j in range(N-1):
                M = torch.ones(t.cores[-1].shape[-1], len(rsets[j])).to(device)
                for n in range(N-1, j, -1):
                    M = torch.einsum('iaj,ja->ia', [t.cores[n][:, rsets[j][:, n-1-j], :], M])
                rinterfaces[j] = M
            t_linterfaces.append(linterfaces)
            t_rinterfaces.append(rinterfaces)
        return t_linterfaces, t_rinterfaces
    t_linterfaces, t_rinterfaces = init_interfaces()

    def evaluate_function(j):  # Evaluate function over Rs[j] x Rs[j+1] fibers, each of size I[j]
        Xs = []
        for k, t in enumerate(tensors):
            V = torch.einsum('ai,ibj,jc->abc', [t_linterfaces[k][j], tensors[k].cores[j], t_rinterfaces[k][j]])
            Xs.append(V.flatten())

        evaluation = f(*Xs)

        if return_info:
            info['Xs'] = torch.cat((info['Xs'], torch.cat([x[:, None] for x in Xs], dim=1).detach().cpu()), dim=0)
            info['shapes'].append([Rs[j], Is[j], Rs[j + 1]])

        V = torch.reshape(evaluation, [Rs[j], Is[j], Rs[j + 1]])
        return V

    cores = []

    # Cross-interpolation formula, left-to-right
    for j in range(0, N-1):

        # Update tensors for current indices
        V = evaluate_function(j)
        V = torch.reshape(V, [-1, V.shape[2]])  # Left unfolding
        # left_locals[j] = np.random.choice(V.shape[0], V.shape[1], replace=False)
        A = V[left_locals[j], :]

        X = lstsq(V.T, A.T)[0].T
        X = torch.reshape(X, [Rs[j], Is[j], Rs[j + 1]])
        cores.append(X)

        # Map local indices to global ones
        local_r, local_i = np.unravel_index(left_locals[j], [Rs[j], Is[j]])
        lsets[j + 1] = np.c_[lsets[j][local_r, :], local_i]
        for k, t in enumerate(tensors):
            t_linterfaces[k][j + 1] = torch.einsum('ai,iaj->aj',
                                                       [t_linterfaces[k][j][local_r, :], t.cores[j][:, local_i, :]])

    # Leave the last core ready
    X = evaluate_function(N-1)
    cores.append(X)

    if return_info:
        return tn.Tensor(cores), info
    else:
        return tn.Tensor(cores)
