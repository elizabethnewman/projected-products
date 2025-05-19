function[G,U] = hosvd3(A,K)
% an inefficient implementation

U = cell(1,3);
G = A;
for i = 1:3
    [u,~,~] = svd(modeUnfold(A,i),'econ');
    U{i}    = u(:,1:K(i));
    G       = modeProduct(G,U{i}',i);
end

end

