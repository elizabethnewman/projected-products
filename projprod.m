function[C] = projprod(A,B,Q)
% projected product for third-order tensors 
%
% Inputs:
%   A  : n1 x p x n3 array
%   B  : p x n2 x n3 array
%   Q  : n3 x p matrix with orthonormal columns
%
% Outputs:
%   C    : A \starQ^H B, n1 x n2 x n3 array
%

% ------------------------------------------- %
% move to transform domain (PART 1) 
A = modeProduct(A,Q');
B = modeProduct(B,Q');

% ------------------------------------------- %
% facewise multiply (PART 2)
C = facewise(A,B);
% ------------------------------------------- %
% return to spatial domain (PART 3)
C = modeProduct(C,Q);

end

