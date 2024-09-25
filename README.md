# Projected Tensor-Tensor Products
Projected tensor-tensor products for multiway data compression


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
TBD
```


