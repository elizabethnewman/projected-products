# Projected Tensor-Tensor Products
A lightweight package to compute projected tensor-tensor products for multiway data compression

# Installation and Setup

This code is compatible with Matlab R2023a or newer. To install, 
```console
git clone https://github.com/elizabethnewman/projected-products.git
```
To setup the code, open an instance of Matlab and change to the working directory
```console
cd projected-products
setupProjectedProducts
```

The script ```setupProjectedProducts.m``` will set the paths for the directory. 

# Quick Start
Below is an example of how to multiply two tensors and compute the tensor SVD under the projected product:

```console
A = randn(3, 4, 5);
B = randn(4, 6, 5);
Q = orth(randn(5));
Q = Q(:,1:3);

C = projprod(A, B, Q);

[U, S, V] = projsvd(A, B, Q);
```

# Citation
```console
@misc{keegan2024projectedtensortensorproductsefficient,
      title={Projected Tensor-Tensor Products for Efficient Computation of Optimal Multiway Data Representations}, 
      author={Katherine Keegan and Elizabeth Newman},
      year={2024},
      eprint={2409.19402},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2409.19402}, 
}
```

The scripts to reproduce the results from the paper are available in the [experimentsFromPaper](https://github.com/elizabethnewman/projected-products/tree/main/experimentsFromPaper) directory.


