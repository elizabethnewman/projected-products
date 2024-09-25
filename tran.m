
function[A] = tran(A,reverseFlag)
% tran
%   compute the transpose of a tensor
% 
% Inputs:
%   A           : n1 x n2 x n3 x ... x n_{d} array
%   reverseFlag : reverse the order of last 2:n_i frontal slices (boolean ,optional, default = false)
%
% Outputs:
%   AT : n2 x n1 x n3 x ... x n_{d} array
%

if nargin == 0, runMinimalExample; return; end

% transpose frontal slices
d = ndims(A);
A = permute(A,[2,1,3:d]);

% reverse (e.g., for t-product)
if exist('reverseFlag','var') && reverseFlag == 1
    sA = size(A);
    for i = 3:d
        colons = repmat({':'},1,i - 1);
        A(colons{:},2:sA(i)) = A(colons{:},sA(i):-1:2);
    end
end

end


function[] = runMinimalExample()

disp('reverseFlag = 0')
A = reshape(1:(5*4*3*2),[5,4,3,2]);
AT = tran(A);

disp('size(A) = ')
disp(size(A));

disp('size(AT) = ')
disp(size(AT));

assert(isequaln(A,tran(tran(A))), 'check implementation of tran with reverseFlag = 0')


disp('reverseFlag = 1')
B = randn(5,4,3,9);
BT = tran(B,1);

disp('size(B) = ')
disp(size(B));

disp('size(BT) = ')
disp(size(BT));

assert(~isequaln(BT,tran(B)), 'check implementation of tran with reverseFlag = 1')

assert(isequaln(A,tran(tran(A,1),1)), 'check implementation of tran with reverseFlag = 1')

end
