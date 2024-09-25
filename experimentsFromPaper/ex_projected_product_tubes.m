% Example from Appendix A

clear; clc;

% form tubes
a = [2;4;6;8];   a = reshape(a,1,1,[]);
b = [1;-1;1;0];  b = reshape(b,1,1,[]);

% form M
H = 0.5 * [1, 1, 1, 1; 
    1, 1, -1, -1; 
    sqrt(2), -sqrt(2), 0, 0; 
    0, 0, sqrt(2), -sqrt(2)];
M = H';

assert(fronorm(M' * M - eye(size(M,1))) < 1e-8)

% partition
p  = 2;
Q1 = M(1:p,:)';
Q2 = M(p+1:end,:)';

% compute product
abM  = mprod(a,b,M);
abQ1 = projprod(a,b,Q1);
abQ2 = projprod(a,b,Q2);

assert(fronorm(abQ1 + abQ2 - abM) / fronorm(abM) < 1e-8)

% show projections lie in null space
aQ1 = modeProduct(a,Q1 * Q1');
aQ2 = modeProduct(a,Q2 * Q2');

bQ1 = modeProduct(b,Q1 * Q1');
bQ2 = modeProduct(b,Q2 * Q2');

assert(fronorm(modeProduct(aQ1,Q2')) < 1e-8)
assert(fronorm(modeProduct(aQ2,Q1')) < 1e-8)
assert(fronorm(modeProduct(bQ1,Q2')) < 1e-8)
assert(fronorm(modeProduct(bQ2,Q1')) < 1e-8)

% show projections zero out frontal slices
abQ1M = modeProduct(abQ1,M);
abQ2M = modeProduct(abQ2,M);

assert(fronorm(abQ1M(:,:,p+1:end)) < 1e-8)
assert(fronorm(abQ2M(:,:,1:p)) < 1e-8)
