function[B] = modeProduct(A,M,k)
% Mode-k product
%   Apply a matrix M along the mode-k fibers of a tensor A
%       A x_k M
%
% Inputs:
%   A  : n1 x n2 x ... x n_{k-1} x n_{k} x n_{k+1} x ... x n_{d} array
%   M  : p x n_k matrix
%   k  : dimension along which to unfold (optional, default k = d)
%
%   NOTE: 'orthoFlag' supersedes 'invFlag' (use cautiously with derivative checks)
%
% Outputs:
%   B    : A x_k M, n1 x n2 x p
%

% parse inputs

if ~exist('k','var') || isempty(k), k = max(ndims(A),3); end


% get output size (will change if M is not square)
sB    = size(A);
sB(k) = size(M,1);

B = modeFold(M * modeUnfold(A,k),sB,k);


end


