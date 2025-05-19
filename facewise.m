function[C] = facewise(A,B)
% facewise product
%   Multiply the frontal slices of tensors
%
% Inputs:
%   A  : n1 x p x n3 x ... x n_{d} array
%   B  : p x n2 x n3 x ... x n_{d} array
%
% Outputs:
%   C    : n1 x n2 x ... x n_{k-1} x p x n_{k+1} x ... x n_{d} array
%

if nargin == 0, runMinimalExample; return; end

% multiply
C = pagemtimes(A,B);

end



function[] = runMinimalExample()
    A = reshape(1:(5*4*3*2),[5,4,3,2]);
    B = reshape(1:(4*7*3*2),[4,7,3,2]);
    C = facewise(A,B);
    
    disp('size(A) = '); disp(size(A));
    disp('size(B) = '); disp(size(B));
    disp('size(C) = '); disp(size(C));
    

end
