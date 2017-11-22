# python-nmf-methods

## About
Repository holding various implementations of specific NMF methods.

As a starting point the code provided implements an inefficient version of the NMF approach described in 
```
Essid, S. and Févotte, C.,
Smooth nonnegative matrix factorization for unsupervised audiovisual document structuring. 
IEEE Transactions on Multimedia, 15(2), 2013, pp.415-425.
```

The paper describes a method for unsupervised NMF on datasets of Bag-of-Words features learned from video and audio data. The method is used for the speaker diarization (who spoke when) problem. The optimization process includes regularization for smoothness and sparsity for the activation matrix.

**Note:** I am aware that I could do with more factorized code and better nesting of loops, however I initially wanted something that works and that can pass my unit tests. The code will be improved over time and implemented in a GPU accelerated framework (i.e. minpy) with MiniBatch support. 

There are currently two functions in ```scnmf.py```. ```SmoothNMF``` is much faster than ```SmoothConvexNMF``` but the later works better for audio.

## How to run

```scnmf.py``` depends on the ```numpy``` python package. You can install it with ```pip install numpy [--user]```

Then you can create a new file ```runnmf.py``` at the same directory as ```scnmf.py``` and add the following:

```python
from scnmf import *
X = np.abs(np.random.randn(10, 3))
Y = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
]).astype(np.float32)
Y += np.abs(np.random.randn(Y.shape[0], Y.shape[1]) * 0.00001)
V = np.matmul(X, Y)

L, H, cost = SmoothConvexNMF(V, 3, beta=0.01, max_iter=1000)

# Originally V = WH, but here we constrained W to be a linear combination of the rows of V.

W = np.matmul(V, L)

print(cost[-1])

# Plot the cost (requires matplotlib)
import matplotlib.pyplot as plt
plt.plt(cost)
plt.title('cost')
plt.xlabel('iteration #')

# Plot dictionary matrix W
plt.imshow(W, aspect='auto')
plt.title('W')

```

You can run it then with ```python runnmf.py``` which will print the cost at the last iteration. Of course you could do better things with it like plotting the cost curve, showing **V**, **W**, and **H** as images, and the like.

