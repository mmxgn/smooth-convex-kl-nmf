import numpy as np

try:
    import tqdm
else:
    # If tqdm is not found, just return the generator
    tqdm = lambda x: x

def smooth(H):
    # Used as regularization for "smoothing" the resulting activations across columns
    return 0.5 * np.sum([(H[:, n] - H[:, n - 1]) ** 2 for n in range(1, H.shape[1])])

def objfunc(V, W, H, beta=0.0):
    # Kullback-Leibler Divergence + "Smoothness" regularizer
    Lamda = np.diag([np.linalg.norm(W[:, k], 1) for k in range(W.shape[1])])
    return np.sum(
        V * np.log(V / np.matmul(W, H)) - V + np.matmul(W, H)
    ) + beta * smooth(np.matmul(Lamda, H))

def update_h_given_w(V, W, H, beta=0.0):
    # Consider W fixed, then update towards a better H. Used in SmoothNMF.
    # beta is the regularization penalizing weight.

    F = V.shape[0]
    N = V.shape[1]
    K = W.shape[1]

    # These are lambda_k in the paper, the norms of the columns of W
    wcolnorms = np.array([[np.linalg.norm(W[:, k], 1) for k in range(K)]]).T

    # The current approximation
    vhat = np.matmul(W, H)

    psi = H * np.matmul(W.T, V / vhat)
    lamda = lambda k: wcolnorms[k]

    if beta == 0:
        Hnew = np.zeros_like(H)
        Hnew = psi / wcolnorms

        return Hnew
    else:
        a = np.zeros((K, N))
        b = np.zeros((K, N))
        for k in range(K):
            for n in range(N):
                if n == 0:
                    b[k, 0] = lamda(k) * (1 - beta * lamda(k) * H[k, 1])
                    a[k, 0] = beta * lamda(k) ** 2
                elif n == N - 1:
                    b[k, N - 1] = lamda(k) * (1 - beta * lamda(k) * H[k, N - 2])
                    a[k, N - 1] = beta * lamda(k) ** 2
                else:
                    a[k, n] = 2 * beta * lamda(k) ** 2
                    b[k, n] = lamda(k) * (1 - beta * lamda(k) * (H[k, n - 1] + H[k, n + 1]))
        Hnew = (np.sqrt(b ** 2 + 4 * a * psi) - b) / (2 * a)
        return Hnew


def update_w_given_h(V, W, H, beta=0.0):
    # Consider H fixed, then update towards a better W. Used in SmoothNMF.
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

def update_l_given_h(V, L, H, beta=0.0):
    # Consider H fixed, then update towards a better L such that W = VL. Used in SmoothConvexNMF.
    # beta is the regularization penalizing weight. The idea behind W = VL is that the dictionary W
    # is chosen from data points from the original data V.

    F = V.shape[0]
    N = V.shape[1]
    M = L.shape[0]
    K = L.shape[1]

    vhat = np.matmul(np.matmul(V, L), H)

    phi = np.zeros((M, K))

    for m in range(M):
        for k in range(K):
            phi[m, k] = L[m, k] * np.sum([
                np.sum([
                    V[f, m] * V[f, n] / vhat[f, n] * H[k, n] for n in range(N)
                ]) for f in range(F)
            ])

    a = np.zeros((M, K))
    b = np.zeros((M, K))
    sigma_k = np.sum(H, axis=1).reshape(1, K)

    s_k = np.zeros((K,))
    delta = np.zeros((M,))
    for k in range(K):
        s_k[k] = 2 * smooth(H[k, :].reshape(1, -1))
    for m in range(M):
        delta[m] = np.sum([V[f, m] for f in range(F)])

    for m in range(M):

        for k in range(K):
            a[m, k] = beta * s_k[k] * delta[m] ** 2
            b[m, k] = (sigma_k[0, k] + beta * np.sum([
                delta[n] * L[n, k] for n in range(N) if n != m
            ])) * delta[m]

    Lnew = (np.sqrt(b ** 2 + 4 * a * phi) - b) / (2 * a)

    return Lnew


def SmoothConvexNMF(V, k, beta=0.001, tol=1e-8, max_iter=100, n_trials_init=10):
    # Smooth and Convex NMF constraints the activations H to be smooth and "sparse"

    # Initialize randomly after some trials
    best_cost = np.inf

    for n in range(n_trials_init):

        L = np.abs(np.random.randn(V.shape[1], k))
        H = np.abs(np.random.randn(k, V.shape[1]))

        W = np.matmul(V, L)
        cost = objfunc(V, W, H, beta)

        if cost < best_cost:
            Linit = L
            Hinit = H
            best_cost = cost

    Lh = Linit
    Hh = Hinit

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


def SmoothNMF(V, k, beta=0.0, tol=1e-8, max_iter=100, n_trials_init=10):
    # SmoothNMF constraints the activations to be "smooth"

    # Initialize randomly after some trials
    best_cost = np.inf

    for n in range(n_trials_init):

        W = np.abs(np.random.randn(V.shape[0], k))
        H = np.abs(np.random.randn(k, V.shape[1]))
        cost = objfunc(V, W, H, beta)
        if cost < best_cost:
            Winit = W
            Hinit = H
            best_cost = cost

    Wh = Winit
    Hh = Hinit

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

        Hh = update_h_given_w(V, Wh, Hh, beta)
        Wh = update_w_given_h(V, Wh, Hh, beta)

    return Wh, Hh, costs[:I]