import sys; sys.path.append('../')
import culpable as cp

import numpy as np


Ds = np.array([0.1, 0.5, 1., 1.5, 2.5, 5., 10.])



def test_M_from_D_bw():
    Ms = [5.8, 6.5968258, 6.94, 7.14074404, 7.39365161, 7.7368258, 8.08]

    _Ms = cp.magnitudes.M_from_D(Ds, ref='BW_2006')
    
    for i, M in enumerate(Ms):
        assert np.isclose(M, _Ms[i])

