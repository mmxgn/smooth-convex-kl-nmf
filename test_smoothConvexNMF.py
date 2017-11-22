from unittest import TestCase
from scnmf import smoothConvexNMF, objfunc
import numpy as np


class TestSmoothConvexNMF(TestCase):
    def test_nonincreasing_cost(self):
        # Start a random 7x4 matrix
        V = np.abs(np.random.randn(7, 4))
        L, H, cost = smoothConvexNMF(V, 3, beta=0.001)

        for n in range(1, len(cost)):
            if cost[n] > cost[n-1]:
                self.fail("Cost increasing, something went wrong.")

    def test_almost_equal(self):
        # Start with a known V combination
        X = np.abs(np.random.randn(10, 3))
        Y = np.array([
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        ]).astype(np.float32)
        Y += np.abs(np.random.randn(Y.shape[0], Y.shape[1]) * 0.00001)
        V = np.matmul(X, Y)

        L, H, cost = smoothConvexNMF(V, 3, beta=0.01, max_iter=1000)
        if np.mean((V - np.matmul(np.matmul(V, L), H))**2) > 0.1:
            self.fail("Mean Squared Error is more than it should")
        if objfunc(V, np.matmul(V, L), H, beta=0.01) > 10:
            self.fail("K-L divergence is more than it should")
