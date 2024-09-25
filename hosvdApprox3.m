function[A] = hosvdApprox3(G,U,K)
% an inefficient implementation

if ~exist('K','var') || isempty(K)
    K = cellfun(@(x) size(x,2),U,'UniformOutput',true);
end

A = G;
for i = 1:3
    A = modeProduct(A,U{i}(:,1:K(i)),i);
end

end

