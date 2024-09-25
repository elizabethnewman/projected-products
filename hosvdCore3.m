
function[G] = hosvdCore3(A,U,K)
G = A;
for i = 1:3
    G = modeProduct(G,U{i}(:,1:K(i))',i);
end

end