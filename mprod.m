function[C] = mprod(A,B,M)
% projected product for third-order tensors 
%
% Inputs:
%   A  : n1 x p x n3 array
%   B  : p x n2 x n3 array
%   M  : n3 x n3 unitary matrix
%
% Outputs:
%   C    : A \starM B, n1 x n2 x n3 array
%

% ------------------------------------------- %
% move to transform domain (PART 1) 
A = modeProduct(A,M);
B = modeProduct(B,M);

% ------------------------------------------- %
% facewise multiply (PART 2)
C = facewise(A,B);
% ------------------------------------------- %
% return to spatial domain (PART 3)
C = modeProduct(C,M');

end

