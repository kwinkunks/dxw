# coding: utf-8
# Author: Matt Hall and others, see below
# Email: matt@agilescientific.com
# Licence: LGPL

# Dynamic {Time, Depth, Whatever} Warping
# A simple DXW algorithm, originally from Sakoe & Chiba 1978,
# https://www.irit.fr/~Julien.Pinquier/Docs/TP_MABS/res/dtw-sakoe-chiba78.pdf.
# This cost matrix algorithm was adapted from this blog post by 
# Abhishek Mishra](https://medium.com/walmartglobaltech/time-series-similarity-using-dynamic-time-warping-explained-9d09119e48ec),
# and [this other one by Jeremy Zhang](https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd). These are both very similar to the pseudocode on [the _Dynamic time warping_ Wikipedia page](https://en.wikipedia.org/wiki/Dynamic_time_warping).
#
import numpy as np

def get_cost(s1, s2):
    """
    Make the cost matrix. Very basic algorithm, no windowing.

    Args:
        s1 (array): Signal 1.
        s2 (array): Signal 2.

    Returns:
        ndarray: The cost.

    Examples:
    >>> s1 = [1, 1, 3, 1, 1, 3, 1, 1]
    >>> s2 = [1, 3, 1, 3, 1]
    >>> cost = get_cost(s1, s2)
    >>> cost.shape
    (8, 5)
    """
    if (len(s1) == 0) or (len(s2) == 0):
        raise ValueError("Signals cannot have 0 length.")

    # Set the initial cost to infinity, except the start.
    L1, L2 = len(s1) + 1, len(s2) + 1
    cost = np.zeros((L1, L2)) + np.inf
    cost[0, 0] = 0
    
    # This is faster than ndindex or ndenumerate.
    for i in range(1, L1):
        for j in range(1, L2):
            prev_min = np.min([cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]])
            cost[i, j] = prev_min + abs(s1[i - 1] - s2[j - 1])

    return cost[1:, 1:]


def backtrack(cost):
    """
    Adapted from page 9 of:
    https://seninp.github.io/assets/pubs/senin_dtw_litreview_2008.pdf
    """
    i, j = np.array(cost.shape) - 1
    path = [[i, j]]
    while (i > 0) or (j > 0):
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if cost[i-1, j] == min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1]):
                i -= 1
            elif cost[i, j-1] == min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1]):
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append([i, j])
    return np.array(path[::-1])


def dxw(s1, s2):
    """
    The path and cost matrix between two signals a and b.
    """
    cost = get_cost(s1, s2)
    return backtrack(cost), cost
