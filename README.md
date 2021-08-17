# (Soft-)DTW for JAX

Dynamic Time Warping (DTW, [1]) calculates the distance between two time series by finding the
optimal alignment of points in both timeseries and calculating the distance based on this alignment.
It can be calculated efficiently using dynamic programming [1], which is what this implementation uses as well.

This idea is extended in Soft-DTW [2],
which replaces the minimum operator by a soft-minimum operator,
in order to make the distance function differentiable everywhere.
Therefore, Soft-DTW is well-suited as a loss function for neural networks.

This repository contains JAX implementations of both DTW and Soft-DTW,
which are compatible with the JAX transformations like `grad`, `jit`, `vmap`, etc.


## References

1. H. Sakoe, S. Chiba.
   *'Dynamic programming algorithm optimization for spoken word recognition'*,
   IEEE Trans. Acoust., Speech, Signal Process., 1978.

2. M. Cuturi, M. Blondel.
   *'Soft-DTW: a Differentiable Loss Function for Time-Series'*,
   Proc. ICML, 2017.
   [arxiv](https://arxiv.org/abs/1703.01541)

This implementation was inspired by
[mblondel/soft-dtw](https://github.com/mblondel/soft-dtw)
and [Maghoumi/pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda).
