clear; clc;

saveDir  = './results_rank1_frontal_slices/';
fileName = 'results';
if ~exist(saveDir, 'dir'), mkdir(saveDir); end

%% construct tensor
% helper functions
twist   = @(x) reshape(x,size(x,1),1,[]);
squeeze = @(x) reshape(x,size(x,1),[]);

% for reproducibility
rng(42);

% choose sizes
n  = 16;
n3 = 8;

% orthogonal factor tensors/matrices
U = orth(randn(n,n3));
V = orth(randn(n,n3));

% construct A
A = facewise(twist(U),tran(twist(V)));

%% verify mode unfolding formulas

% mode-1
E = zeros(n3^2,n3);
for i = 1:n3
    % i-th standard basis vector
    ei      = zeros(n3,1);
    ei(i)   = 1;

    % matrix E
    E((i-1)*n3+1:i*n3,:) = diag(ei);
end

% orthogonality of E
tol = 1e-8;
assert(fronorm(E' * E - eye(n3)) / fronorm(eye(n3)) < tol)

% mode-1 unfolding formula
A1 = U * (E' * kron(eye(n3),V'));
assert(fronorm(A1 - modeUnfold(A,1)) / fronorm(A) < tol)

% mode-2 unfolding formula
A2 = V * (E' * kron(eye(n3),U'));
assert(fronorm(A2 - modeUnfold(A,2)) / fronorm(A) < tol)

% mode-3 unfolding formula
UTilde = zeros(n3,n*n3);
VTilde = zeros(n*n3,n^2);
for i = 1:n
    UTilde(:,(i-1)*n3+1:i*n3) = diag(U(i,:));
    VTilde(:,(i-1)*n+1:i*n)   = kron(eye(n),V(i,:)');
end

A3 = UTilde * VTilde;
assert(fronorm(A3 - modeUnfold(A,3)) / fronorm(A) < tol)

% orthogonality of UTilde and VTilde
assert(fronorm(UTilde * UTilde' - eye(n3)) / fronorm(eye(n3)) < tol)
assert(fronorm(VTilde * VTilde' - eye(n * n3)) / fronorm(eye(n * n3)) < tol)


% verify Q-product formula
rng(123);
p = 4;
Q = orth(randn(n3,p));

ATilde = zeros(n,n,p);
for i = 1:p
    ATilde(:,:,i) = U * diag(Q(:,i)) * V';
end

AHat = modeProduct(A,Q');

assert(fronorm(AHat - ATilde) / fronorm(AHat) < tol)

%% error vs. compression

if ~exist([saveDir,'tsvd'], 'dir'), mkdir([saveDir,'tsvd']); end

for QName = {'I','Z','C','H','H4','H2'}
    if ~exist([saveDir,'tsvd/',QName{1}], 'dir'), mkdir([saveDir,'tsvd/',QName{1}]); end

    switch QName{1}
        case 'Z'
            % compute mode-3 left singular matrix as transformation
            [Q,~,~] = svd(modeUnfold(A,3),'econ');
        case 'I'
            Q = eye(n3);
        case 'C'
            Q = dctmtx(n3)';
        case 'H8'
            Q = generate_haar(n3)';

        case 'H4'
            H = generate_haar(4);
            Q = kron(eye(floor(n3 / 4)),H');
        case 'H2'
            H = generate_haar(2);
            Q = kron(eye(floor(n3 / 2)),H');
    end
    
    % compute Q-SVD error and compression for various parameter choices
    [error,store] = qsvdErrorStore(A,Q);
    
    T = [kron(ones(1,n3),1:n3)',kron(1:n3,ones(1,n3))',...
        error(:),error(:) / fronorm(A), ...
        store(:), numel(A) ./ store(:)];

    T = array2table(T,'VariableNames',{'k','p','err','rel_err','comp','rel_comp'});
    writetable(T,[saveDir,'tsvd/',QName{1},'/results.csv'])
end


%% HOSVD experiment
if ~exist([saveDir,'hosvd'], 'dir'), mkdir([saveDir,'hosvd']); end
% compute HOSVD error and compression for various parameter choices
error = zeros(n3,n3,n3);
store = zeros(n3,n3,n3);

% form full HOSVD basis
[~,U] = hosvd3(A,size(A));

for k1 = 1:n3
    for k2 = 1:n3
        for k3 = 1:n3
            % form core
            G = hosvdCore3(A,U,[k1,k2,k3]);

            % form approximation
            Ak1k2k3 = hosvdApprox3(G,U,[k1,k2,k3]);

            % compute error and storage
            error(k1,k2,k3) = fronorm(A - Ak1k2k3);
            store(k1,k2,k3) = n3 * k3 + n * k1 + n * k2 + k1 * k2 * k3;

        end
    end
end

T = [kron(kron(ones(1,n3),ones(1,n3)),1:n3)',...
    kron(kron(ones(1,n3),1:n3),ones(1,n3))',...
    kron(kron(1:n3,ones(1,n3)),ones(1,n3))',...
    error(:),error(:) / fronorm(A), ...
    store(:), numel(A) ./ store(:)];

T = array2table(T,'VariableNames',{'k1','k2','k3','err','rel_err','comp','rel_comp'});
writetable(T,[saveDir,'hosvd/results.csv'])

%% statistical study of Q-SVD storage with random orthogonal transformations

if ~exist([saveDir,'tsvd/W'], 'dir'), mkdir([saveDir,'tsvd/W']); end

rng(123);
nTrials = 100;

E = [];
S = [];
errNames = {};
compNames = {};
for i = 1:nTrials
    W = orth(randn(n3));
    % W = kron(eye(floor(n3 / size(W,1))),W);
    % tmp = randn(n3);
    % W = expm(1e1 * (tmp - tmp'));

    [error,store] = qsvdErrorStore(A,W);

    E = cat(2,E,error(:) / fronorm(A));
    S = cat(2,S,numel(A) ./ store(:));

    errNames  = cat(2,errNames,sprintf('rel_err_%d',i));
    compNames = cat(2,compNames,sprintf('rel_comp_%d',i));
end

mE      = mean(E,2);
stdE    = std(E,[],2);
mS      = mean(S,2);
stdS    = std(S,[],2);

zScore = 1.96; % 95% confidence interval
confE  = zScore * stdE / sqrt(nTrials);
confS  = zScore * stdS / sqrt(nTrials);

T = array2table([kron(ones(1,n3),1:n3)',kron(1:n3,ones(1,n3))', ...
    mE,stdE,mE+confE,mE-confE,min(E,[],2),max(E,[],2),mS,stdS,mS+confS,mS-confS,min(S,[],2),max(S,[],2),E,S],...
    'VariableNames',[{'k','p','rel_err_mean','rel_err_std', 'rel_err_plus', 'rel_err_minus','rel_err_min','rel_err_max','rel_comp_mean','rel_comp_std', 'rel_comp_plus', 'rel_comp_minus','rel_comp_min','rel_comp_max'},errNames,compNames]);
writetable(T,[saveDir,'tsvd/W/results.csv'])


%% create visualizations

% read in results
QSVD  = readtable('results_rank1_frontal_slices/tsvd/H4/results.csv');
HOSVD = readtable('results_rank1_frontal_slices/hosvd/results.csv');


% create plot
figure(1); clf;

cmap = get(gca,'ColorOrder');

count = 1;
for p = n3
    idx = (QSVD.p == p);
    loglog(QSVD.rel_err(idx), QSVD.rel_comp(idx),'o-','Color',cmap(count,:))
    
    hold on;
    idx = (HOSVD.k3 == p);
    loglog(HOSVD.rel_err(idx),HOSVD.rel_comp(idx),'*','Color',cmap(count,:))

    count = count + 1;
end

xlim([1e-1,1e0])



% idx = QSVD.p == 1
% loglog(E(1:n3,:) / fronorm(A), numel(A) ./ S(1:n3,:))

QSVDRandom = readtable('results_rank1_frontal_slices/tsvd/W/results.csv');

figure(1); clf;

cmap = get(gca,'ColorOrder');

count = 1;
for p = [2,3]
    idx = (QSVDRandom.p == p);
    
    % statistics
    m        = 14; % number of headers before each experimental result
    nTrials  = floor((size(QSVDRandom,2) - m) / 2);
    trialIdx = (m+1):(m + nTrials);

    % extract relevant data
    E    = table2array(QSVDRandom(:,trialIdx));
    mE   = QSVDRandom.rel_err_mean(idx);
    stdE = QSVDRandom.rel_err_std(idx);

    S    = table2array(QSVDRandom(:,trialIdx(end)+1:end));
    mS   = QSVDRandom.rel_comp_mean(idx);
    stdS = QSVDRandom.rel_comp_std(idx);

    % fill plot
    n = 1;
    fill([mE-n*stdE;flipud(mE+n*stdE)],[mS-n*stdS;flipud(mS+n*stdS)],cmap(count,:),'FaceAlpha',0.5)

    hold on;
    loglog(E(idx,:), S(idx,:),'-','Color',[0,0,0,0.25]);


    % standard deviation
    loglog(mE,mS,'k-','LineWidth',2,'Color',[0,0,0])
    for j = 1:n
        loglog(mE+j*stdE,mS+j*stdS,'-','LineWidth',2,'Color',[0,0,0,0.75])
        loglog(mE-j*stdE,mS-j*stdS,'-','LineWidth',2,'Color',[0,0,0,0.5])
    end

    idx = (HOSVD.k3 == p);
    loglog(HOSVD.rel_err(idx),HOSVD.rel_comp(idx),'*','Color',cmap(count,:))

    count = count + 1;
end

%% plot ratios

T1 = readtable('results_rank1_frontal_slices/tsvd/I/results.csv');
T2 = readtable('results_rank1_frontal_slices/hosvd/results.csv');

% fix k and p, find all HOSVD cases 
%   - which produce no worse relative error, and
%   - which offer no less compression
% compute the compression ratios

ratioRE = [];
for p = 1:n3
    for k = 1:n3
        idx1 = find((T1.p == p) .* (T1.k == k)); % a single value
        re1  = T1(idx1,:).rel_err;
        cr1  = T1(idx1,:).comp;

        % HOSVD 
        idx2 = find((T2.rel_err <= re1));
        cr2  = T2(idx2,:).comp;

        r = cr2 / cr1;
        
        tmp1 = kron(ones(length(idx2),1),[k,p]);
        tmp2 = [T2(idx2,:).k1, T2(idx2,:).k2, T2(idx2,:).k3];
        ratioRE = cat(1,ratioRE,[tmp1,tmp2,r]);
    end
end

% R = [];
% for p = 1:n3
%     tmp1 = T1(T1.p == 4,:);
%     tmp2 = T2(T2.k3 == 4,:);
%     for k = 1:n3
%         idx1 = find((T1.p == p) .* (T1.k == k));
% 
%         for k1 = 1:n3
%             for k2 = 1:n3
% 
%                 idx2 = find((T2.k1 == k1) + (T2.k2 == k));
% 
%                 tmp = kron([k,k1,k2,p],ones(length(idx2)));
%                 R = cat(1,R,[tmp,T2(idx2,:).comp / T1(idx1,:).comp]);
%             end
%         end
% 
%     end
% end

%% helper functions

function[error,store] = qsvdErrorStore(A,Q)

n  = size(A,1);
n3 = size(A,3);

error = zeros(n3);
store = zeros(n3);

for k = 1:n3
    for p = 1:n3
        % compute Q-SVD
        [u,s,v] = projsvd(A,Q(:,1:p),k);

        % form approximation
        Akp = projprod(u,projprod(s,tran(v),Q(:,1:p)),Q(:,1:p));

        % compute error and storage
        error(k,p) = fronorm(A - Akp);
        store(k,p) = n3 * p + 2 * n * p * k;
    end
end

end

