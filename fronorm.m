function[nrmA] = fronorm(A)
% Compute the Frobenius norm of d-dimensional array
%
% Inputs:
%   A : n1 x n2 x ... x n_{d} array
%
% Outputs:
%  nrmA : Frobenius norm of array (scalar)
%

nrmA = norm(A(:));

end

