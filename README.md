# python-nmf-methods
Repository holding various implementation of specific NMF methods.

As a starting point the code provided implements an inefficient version of the NMF approach described in 
```
Essid, S. and Févotte, C.,
Smooth nonnegative matrix factorization for unsupervised audiovisual document structuring. 
IEEE Transactions on Multimedia, 15(2), 2013, pp.415-425.
```

**Note:** I am aware that I could do with more factorized code and better nesting of loops, however I initially wanted something that works and that can pass my unit tests. The code will be improved over time and implemented in a GPU accelerated framework (i.e. minpy) with MiniBatch support. 
