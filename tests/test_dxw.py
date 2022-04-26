import pytest

from dxw import dxw
import numpy as np


def test_empty():    
    with pytest.raises(ValueError):
        path, cost = dxw([1, 3, 1, ], [])

def test_cost_and_path():
    s1 = [1, 1, 3, 1, 1, 3, 1, 1]
    s2 = [1, 3, 1, 3, 1]
    path, cost = dxw(s1, s2)
    assert path.shape == (8, 2)
    np.testing.assert_allclose(cost[0], [0., 2., 2., 4., 4.])
