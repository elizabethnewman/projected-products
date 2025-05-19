function[U,S,V,info] = projsvdII(A, gamma, Q, varargin)

if nargin == 0, runMinimalExample(); return; end
if ~exist('gamma','var'), gamma = 1; end

% parse arguments
for k = 1:2:length(varargin), eval([varargin{k}, ' = varargin{k + 1};']); end
if ~exist('tau','var'), tau = 1e-14; end % slack variable

% compute tsvd, but stay in transform domain
AHat             = modeProduct(A,Q');
[UHat,SHat,VHat] = facewiseSVD(AHat);

% collect singular values
s        = tdiag(SHat);
nrmSHat2 = norm(s,'fro')^2;

% sort
[v,idx] = sort(s(:).^2,'descend');

% find truncation
cutoff = find(cumsum(v) / nrmSHat2 > gamma - tau,1);

% truncate (a loop might be more efficient in terms of storage)
s(idx(cutoff + 1:end)) = 0;

% get multirank and implicit rank
rho    = reshape(sum(s > 0, 1),1,[]);
rhoMax = max(rho);
r      = sum(rho);

% return to spatial domain (can be optional - better to stay in transform domain for storage)
U = modeProduct(UHat(:,1:rhoMax,:),Q);
S = modeProduct(SHat(1:rhoMax,1:rhoMax,:),Q);
V = modeProduct(VHat(:,1:rhoMax,:),Q);

% info
info.rho    = rho;
info.rhoMax = rhoMax;
info.r      = r;
info.gamma  = gamma;

end


function runMinimalExample()

rng(42);

A = randn(10,10,10);

gamma = linspace(0.5,1,50);

Q = orth(randn(size(A,3),2));

err = zeros(1,length(gamma));
r   = zeros(1,length(gamma));
rho = zeros(size(Q,2),length(gamma));
for i = 1:length(gamma)
    [U,S,V,info] = projsvdII(A,gamma(i),Q);
    err(i)  = norm(A - tprod(U,tprod(S,ttran(V))),'fro');
    r(i)    = info.r;
    rho(:,i) = info.rho;
end

nrmA = norm(A,'fro');

figure(2); clf;

subplot(1,3,1);
plot(gamma(:), err(:) / nrmA, '-o', 'LineWidth',2, 'MarkerSize',10)
xlabel('\gamma')
ylabel('||A - A_k||_F / ||A||_F')
set(gca,'FontSize',18)
grid on
title('relative error')

cmap = jet(length(gamma));
subplot(1,3,2);
scatter(gamma(:), r(:),100,cmap,'LineWidth',2)
xlabel('\gamma')
ylabel('r')
set(gca,'FontSize',18)
grid on
title('implicit rank')

subplot(1,3,3);
for i = 1:length(gamma)

    % some cool size tricks

    % get counts 
    p = unique(rho(:,i));
    count = zeros(1,numel(p));
    for k = 1:numel(p)
        count(k) = sum(rho(:,i) == p(k));
    end

    % find most common rank
    [~,idx] = sort(count,'descend');
    % 
    % % give larger size for bigger rank (relatively)
    % s = zeros(size(rho,i));
    % for k = 1:size(rho,1)
    % 
    % end


    scatter(gamma(i) + 0 * rho(:,i), rho(:,i),100, cmap(i,:),'LineWidth',2)
    hold on;
end
xlabel('\gamma')
ylabel('\rho')
set(gca,'FontSize',18)
grid on
hold off;
title('multirank')

end
