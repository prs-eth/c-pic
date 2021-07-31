import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.svd import fortran_reshape, z_reorder, inv_z_reorder
import torch
import numpy as np

def test_fortran_reshape():
    a = torch.rand(2, 1024)
    assert np.allclose(
        a.numpy().reshape([2]*11, order='F'), 
        fortran_reshape(a, [-1] + [2]*10)
    )


def test_z_order():
    a = torch.rand((2, 32, 32))
    assert np.allclose(
        inv_z_reorder(
            z_reorder(a)
        ).detach().cpu().numpy(),
        a.detach().cpu().numpy()
    )
    
    assert np.allclose(
        a.numpy().reshape([2] + [2]*10, order='F').transpose(0, 1, 6, 2, 7, 3, 8, 4, 9, 5, 10),
        z_reorder(a)
    )
    
    
def test_indices():
    # Fortran
    a = 3
    b = 4
    c = 5
    t = np.arange(a * b * c).reshape((a, b, c))
    x = np.random.choice(a)
    y = np.random.choice(b)
    z = np.random.choice(c)
    assert t[x, y, z] == t.reshape(-1, order='F')[a * b * z + a * y + x]

    a = 3
    b = 4
    c = 5
    d = 6
    t = np.random.random(a * b * c * d).reshape((a, b, c, d))
    x = np.random.choice(a)
    y = np.random.choice(b)
    z = np.random.choice(c)
    q = np.random.choice(d)
    assert t[x, y, z, q] == t.reshape(-1, order='F')[a * b * c * q + a * b * z + a * y + x]

    a = 3
    b = 4
    c = 5
    d = 6
    e = 7
    t = np.random.random(a * b * c * d * e).reshape((a, b, c, d, e))
    x = np.random.choice(a)
    y = np.random.choice(b)
    z = np.random.choice(c)
    q = np.random.choice(d)
    k = np.random.choice(e)
    assert t[x, y, z, q, k] == t.reshape(-1, order='F')[a * b * c * d * k + a * b * c * q + a * b * z + a * y + x]
    
    a = 2
    b = 3
    c = 4
    d = 5
    e = 6
    t = np.random.random((a, b * c, d * e))
    t2 = t.reshape((a, b, c, d, e), order='F')
    x = np.random.choice(a)
    y = np.random.choice(b)
    z = np.random.choice(c)
    q = np.random.choice(d)
    k = np.random.choice(e)

    p, s, g = np.unravel_index(a * b * c * d * k + a * b * c * q + a * b * z + a * y + x, (a, b * c, d * e), order='F')
    assert t2[x, y, z, q, k] == t.reshape(-1, order='F')[a * b * c * d * k + a * b * c * q + a * b * z + a * y + x]
    assert t2[x, y, z, q, k] == t[p, s, g]

    # C
    a = 3
    b = 4
    c = 5
    t = np.random.random(a * b * c).reshape((a, b, c))
    x = np.random.choice(a)
    y = np.random.choice(b)
    z = np.random.choice(c)
    assert t[x, y, z] == t.reshape(-1)[b * c * x + c * y + z]

    a = 3
    b = 4
    c = 5
    d = 6
    t = np.random.random(a * b * c * d).reshape((a, b, c, d))
    x = np.random.choice(a)
    y = np.random.choice(b)
    z = np.random.choice(c)
    q = np.random.choice(d)
    assert t[x, y, z, q] == t.reshape(-1)[b * c * d * x + c * d * y + d * z + q]

    a = 3
    b = 4
    c = 5
    d = 6
    e = 7
    t = np.random.random(a * b * c * d * e).reshape((a, b, c, d, e))
    x = np.random.choice(a)
    y = np.random.choice(b)
    z = np.random.choice(c)
    q = np.random.choice(d)
    k = np.random.choice(e)
    assert t[x, y, z, q, k] == t.reshape(-1)[b * c * d * e * x + c * d * e * y + d * e * z + e * q + k]
    

if __name__ == '__main__':
    test_fortran_reshape()
    test_z_order()
    test_indices()