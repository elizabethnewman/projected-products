function[U,S,V] = facewiseSVD(A,k)
%facewiseSVD
%   Compute the svd of frontal slices of a tensor
%
% Inputs: 
%   A : n1 x n2 x n3 x ... x n_{d} array
%   k : optional, "econ" (k = 0) or truncation integer (k > 0)
% 
% Outputs: for "econ", p = min(n1,n2), for k > 0, p = k, otherwise, full svd sizes for first two dimensions
%   U : left singular lateral slices, n1 x p x n3 x ... x n_{d} array
%   S : singular tubes, p x p x n3 x ... x n_{d} array
%   V : right singular lateral slilces, n2 x p x n3 x ... x n_{d} array
%

if nargin == 0, runMinimalExample; return; end

% compute svd (always economic for now)
[U,S,V] = pagesvd(A,"econ");


% truncate, if applicable
if exist('k','var') && ~isempty(k) && isnumeric(k) && k > 0
    colons = repmat({':'},1,ndims(A)-2);
    U = U(:,1:k,colons{:});
    S = S(1:k,1:k,colons{:});
    V = V(:,1:k,colons{:});
end

end


function[] = runMinimalExample()
A = randn(4,6,3);
M = randn(3);

% check if M is inverible 
if cond(M) > 1e12, error('M close to singular, bad initialization, re-run test'); end


disp('FULL RANK')
[U,S,V] = tSVDM(A,M);

disp('size(A) = '); disp(size(A));
disp('size(U) = '); disp(size(U));
disp('size(S) = '); disp(size(S));
disp('size(V) = '); disp(size(V));


disp('ECONOMIC')
[U,S,V] = tSVDM(A,M,"econ");

disp('size(A) = '); disp(size(A));
disp('size(U) = '); disp(size(U));
disp('size(S) = '); disp(size(S));
disp('size(V) = '); disp(size(V));

disp('LOW RANK')
[U,S,V] = tSVDM(A,M,1);

disp('size(A) = '); disp(size(A));
disp('size(U) = '); disp(size(U));
disp('size(S) = '); disp(size(S));
disp('size(V) = '); disp(size(V));

end




