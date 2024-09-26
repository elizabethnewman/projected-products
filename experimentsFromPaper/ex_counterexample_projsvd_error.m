clear; clc;


A(:,:,1) = [1,0;0,1];
A(:,:,2) = [1,0;0,-1];

n3 = size(A,3);
k  = 1;
p  = 1;

I = eye(n3);
H = 1 / sqrt(2) * [1, 1; 1, -1];

Q = {'I', I, 'H', H'};

for i = 1:2:length(Q)
    Qp      = Q{i+1}(:,1:p);
    [U,S,V] = projsvd(A,Qp,k);
    Ak      = projprod(U,projprod(S,tran(V),Qp),Qp);
    
    err = fronorm(A - Ak);
    fprintf('||A - A_k(%s)||_F = %0.4f\n',Q{i},err)
end