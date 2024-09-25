function[A] = modeUnfold(A,k)
% Mode-k unfold
%   Turns a tensor into a matrix whose columns are the mode-k fibers
%
% Inputs:
%   A  : n1 x n2 x ... x n_{k-1} x n_{k} x n_{k+1} x ... x n_{d} array
%   k  : dimension along which to unfold (optional, default k = d)
%
% Outputs:
%   A    : n_{k} x (n1 n2 ... n_{k-1} n_{k+1} ... n_{d}) matrix
% 

if nargin == 0, runMinimalExample; return; end

% dimension to unfold 
d = ndims(A);
if ~exist('k','var') || isempty(k), k = d; end

% force first dimension to be k (many correct ways to do this)
idx = [k,1:(k-1),(k+1):d];
A   = permute(A,idx);

% matricize [A(:,:,1), A(:,:,2), ..., A(:,:,d)]
A = A(:,:);


end


function[] = runMinimalExample()
A = reshape(1:120,[5,4,3,2]);
disp('Tensor A')
disp(A);

disp('mode-1 Unfolding: A_{(1)}')
disp(modeUnfold(A,1));

disp('mode-2 Unfolding: A_{(2)}')
disp(modeUnfold(A,2));

disp('mode-3 Unfolding: A_{(3)}')
disp(modeUnfold(A,3));

disp('mode-4 Unfolding: A_{(4)}')
disp(modeUnfold(A,4));

end

