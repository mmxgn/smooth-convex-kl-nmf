try:
    # If minpy is available, use it.
    import minpy.numpy as np
except ImportError:
    import numpy as np
import tqdm


def smooth(H):
    # Used as regularization for "smoothing" the resulting activations across columns
    return 0.5 * np.sum((H[:, :-1] - H[:, 1:]) ** 2)


def smooth_rows(H):
    # Same as above, but smooths the rows instead
    return 0.5 * np.sum((H[:, :-1] - H[:, 1:]) ** 2, axis=1)


def objfunc(V, W, H, beta=0.0):
    # Kullback-Leibler Divergence + "Smoothness" regularizer
    Lamda = np.diag([np.linalg.norm(W[:, k], 1) for k in range(W.shape[1])])
    return np.sum(
        V * np.log(V / np.matmul(W, H)) - V + np.matmul(W, H)
    ) + beta * smooth(np.matmul(Lamda, H))


def update_h_given_w(V, W, H, beta=0.0, lamda=None):
    F = V.shape[0]
    N = V.shape[1]
    K = W.shape[1]

    if lamda is None:
        # Lamda can be precomputed once
        lamda = np.linalg.norm(W, 1, axis=0)
    lamda_col = lamda.reshape(-1, 1)
    vhat = np.matmul(W, H)

    psi = H * np.matmul(W.T, V / vhat)

    if beta == 0:
        Hnew = psi / wcolnorms
        return Hnew
    else:
        a = np.zeros((K, N))
        b = np.zeros((K, N))

        # Edge cases
        b[:, 0] = lamda * (1 - beta * lamda * H[:, 1])
        a[:, 0] = a[:, N - 1] = beta * lamda ** 2
        a[:, 1:N - 1] = 2 * beta * lamda_col ** 2
        b[:, N - 1] = lamda * (1 - beta * lamda * H[:, N - 2])

        # Rest of cases

        Dh = (H[:, :-2] + H[:, 2:])
        b[:, 1:-1] = lamda_col * (1 - beta * lamda_col * Dh)

        Hnew = (np.sqrt(b ** 2 + 4 * a * psi) - b) / (2 * a)
        return Hnew


def update_w_given_h(V, W, H, beta=0.0):
    # Consider H fixed, then update towards a better W. Used in smoothNMF.
    # beta is the regularization penalizing weight.

    F = V.shape[0]
    N = V.shape[1]
    K = W.shape[1]
    vhat = np.matmul(W, H)
    phi = W * np.matmul(V / vhat, H.T)

    sigma_k = np.sum(H, axis=1).reshape(1, K)

    if beta == 0:

        Wnew = phi / sigma_k
        return Wnew
    else:
        a = np.zeros((F, K))
        b = np.zeros((F, K))
        for f in range(F):
            for k in range(K):
                s_k = 2 * smooth(H[k, :].reshape(1, -1))
                a[f, k] = beta * s_k
                b[f, k] = sigma_k[0, k] + beta * s_k * np.sum(W[np.arange(F) != f, k])

        Wnew = (np.sqrt(b ** 2 + 4 * a * phi) - b) / (2 * a)
        return Wnew


def update_l_given_h(V, L, H, beta=0.001):
    F = V.shape[0]
    N = V.shape[1]
    M = L.shape[0]
    K = L.shape[1]

    vhat = np.matmul(np.matmul(V, L), H)

    # The following computes the inner sum V^v_{kn} = \sum_{fn}{v_{fn}/\hat{v}_{fn}*f_{kn}
    VV = np.matmul(V / vhat, H.T)

    # Computes phi as phi_{mk}
    phi = L * np.matmul(V.T, VV)

    sigma_k = np.sum(H, axis=1).reshape(1, K)
    s_k = smooth_rows(H).reshape(1, -1)

    delta = np.sum(V, axis=0).reshape(-1, 1)
    a = beta * np.matmul(delta ** 2, s_k)
    DL = delta * L

    sumColsDL = np.sum(DL, 1).reshape(-1, 1)
    sumDL = np.sum(sumColsDL)
    sigma_k_delta = np.matmul(delta, sigma_k)

    b = np.zeros((M, K))
    b += sumDL - sumColsDL
    b *= beta * delta
    b += sigma_k_delta

    Lnew = (np.sqrt(b ** 2 + 4 * a * phi) - b) / (2 * a)

    return Lnew


def smoothConvexNMF(V, k, beta=0.001, tol=1e-8, max_iter=100, n_trials_init=10, init='random'):
    # Smooth and Convex NMF constraints the activations H to be smooth and "sparse"

    if init == 'random':
        # Initialize randomly after some trials
        best_cost = np.inf

        for n in range(n_trials_init):

            L = np.abs(np.random.randn(V.shape[1], k))
            H = np.abs(np.random.randn(k, V.shape[1]))

            W = np.matmul(V, L)
            cost = objfunc(V, W, H, beta)

            if cost < best_cost:
                Lh = L
                Hh = H
                best_cost = cost
    else:
        # If init is not 'random' then use predefined matrices
        Lh = init['L']
        Hh = init['H']

    costs = np.zeros((max_iter,))
    last_cost = np.inf

    for I in tqdm.tqdm(range(max_iter)):
        cur_cost = objfunc(V, np.matmul(V, Lh), Hh, beta)

        cost_diff = np.abs(cur_cost - last_cost)
        if cost_diff <= tol:
            break

        last_cost = cur_cost

        if I > 0:
            costs[I - 1] = last_cost

        Hh = update_h_given_w(V, np.matmul(V, Lh), Hh, beta)
        Lh = update_l_given_h(V, Lh, Hh, beta)

    return Lh, Hh, costs[:I]


def miniBatchSmoothConvexNMF(V, k, batch_size=5, epochs=1000, beta=0.001, tol=1e-8, sort=True, init='random'):
    best_cost = np.inf

    costs = []
    batchindices = np.array_split(np.arange(V.shape[1]), V.shape[1] / batch_size)

    if init == 'random':
        # Initialize H, L randomly
        H = np.abs(np.random.randn(k, V.shape[1]))
        L = np.abs(np.random.randn(V.shape[1], k))
    else:
        # If init is not 'random' then use predefined matrices
        H = init['H']
        L = init['L']

    for epoch in tqdm.tqdm(range(epochs)):
        W = np.matmul(V, L)
        lamda = np.linalg.norm(W, 1, axis=0)

        for n, batchidx in enumerate(batchindices):
            # Update the activations for each batch
            if n > 0 and n < len(batchindices) - 1:
                H[:, batchidx] = update_h_given_w(V[:, min(batchidx) - 1:max(batchidx) + 2], W,
                                                  H[:, min(batchidx) - 1:max(batchidx) + 2], beta, lamda=lamda)[:, 1:-1]
            if n == 0:
                H[:, batchidx] = update_h_given_w(V[:, min(batchidx):max(batchidx) + 2], W,
                                                  H[:, min(batchidx):max(batchidx) + 2], beta, lamda=lamda)[:, :-1]
            if n == len(batchindices) - 1:
                H[:, batchidx] = update_h_given_w(V[:, min(batchidx) - 1:max(batchidx) + 1], W,
                                                  H[:, min(batchidx) - 1:max(batchidx) + 1], beta, lamda=lamda)[:, 1:]

        # Update the dictionary once per epoch
        L = update_l_given_h(V, L, H, beta)

        cur_cost = objfunc(V, np.matmul(V, L), H, beta)
        costs.append(cur_cost)
    return L, H, costs


def smoothNMF(V, k, beta=0.0, tol=1e-8, max_iter=100, n_trials_init=10):
    # smoothNMF constraints the activations to be "smooth"

    # Initialize randomly after some trials
    best_cost = np.inf

    for n in range(n_trials_init):

        W = np.abs(np.random.randn(V.shape[0], k))
        H = np.abs(np.random.randn(k, V.shape[1]))
        cost = objfunc(V, W, H, beta)
        if cost < best_cost:
            Wh = W
            Hh = H
            best_cost = cost

    costs = np.zeros((max_iter,))
    last_cost = np.inf
    for I in tqdm.tqdm(range(max_iter)):
        cur_cost = objfunc(V, Wh, Hh, beta)

        cost_diff = np.abs(cur_cost - last_cost)
        if cost_diff <= tol:
            break

        last_cost = cur_cost

        if I > 0:
            costs[I - 1] = last_cost

        Hh = update_h_given_w(V, Wh, Hh, beta)
        Wh = update_w_given_h(V, Wh, Hh, beta)

    return Wh, Hh, costs[:I]


