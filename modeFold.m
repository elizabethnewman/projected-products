function[A] = modeFold(A,sA,k)
% Mode-k fold
%   Turns a matrix into a tensor whose mode-k fibers come from the columns
%
% Inputs:
%   A  : n_{k} x (n1 n2 ... n_{k-1} n_{k+1} ... n_{d}) matrix
%   sA : size to reshape A ([n1 x n2 x ... x n_{k-1} x n_{k} x n_{k+1} x ... x n_{d}])
%   k  : dimension along which to fold (optional, default k = d)
%
% Outputs:
%   A    : n1 x n2 x ... x n_{k-1} x n_{k} x n_{k+1} x ... x n_{d} array
%

if nargin == 0, runMinimalExample; return; end


d = length(sA);

if ~exist('k','var') || isempty(k), k = d; end



% undo original unfold
idx = [k,1:(k-1),(k+1):d];
A   = reshape(A,sA(idx));

% permute indices such that mode-1 becomes mode-k
idx = [2:k,1,(k+1):d];
A   = permute(A,idx);

end


function[] = runMinimalExample()

A = reshape(1:24,[3,8]);
disp('Matrix A')
disp(A);

disp('mode-1 Folding')
disp(modeFold(A,[3,4,2],1));


disp('mode-2 Folding')
disp(modeFold(A,[4,3,2],2));

disp('mode-3 Folding')
disp(modeFold(A,[4,2,3],3));

end
