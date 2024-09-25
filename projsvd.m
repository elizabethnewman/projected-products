function[U,S,V] = projsvd(A,Q,k)
% Compute the projected t-SVDM using the starQ-product for third-order tensors
%
% Inputs:
%   A: third-order tensor of size n1 x n2 x n3
%   Q: n3 x p matrix with orthonormal columns 
%   k (optional): integer, truncation parameter 
%
% Outputs:
%   U: n1 x k x n3 tensor containing basis
%   S: k x k x n3 tensor containing singular tubes
%   V: n2 x k x n3 tensor containing normalized coefficients
%      

% set default
if ~exist('Q','var') || isempty(Q), Q = 1; end  % identity
if ~exist('k','var'), k = min(size(A,1),size(A,2)); end

% apply transformation to tubes
A = modeProduct(A,Q');

% facewise svd
[U,S,V] = facewiseSVD(A,k);

% return to spatial domain
U = modeProduct(U,Q);
S = modeProduct(S,Q);
V = modeProduct(V,Q);

end



