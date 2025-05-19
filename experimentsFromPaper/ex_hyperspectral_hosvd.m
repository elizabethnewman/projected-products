
clear; clc;

%% load data

hcube = hypercube('indian_pines.dat');
A     = hcube.DataCube;

% normalize
A = A / fronorm(A);

saveDir  = './results_hyperspectral_hosvd/';
fileName = 'results';
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
if ~exist([saveDir,'orig'], 'dir'), mkdir([saveDir,'orig']); end

% get sizes
[n1,n2,n3]  = size(A);
nrmA        = fronorm(A);
storeA      = n1 * n2 * n3;

%% compute projected tensor SVD with Q = Z (repeat of previous experiment)

[Z,~,~] = svd(modeUnfold(A,3),'econ');
Q = Z;

headers = {'k','p','err','rel_err','comp_ratio','rel_comp_ratio'};

if ~exist([saveDir,'tsvd/'], 'dir'), mkdir([saveDir,'tsvd/']); end
if ~exist([saveDir,'tsvd/img'], 'dir'), mkdir([saveDir,'tsvd/img']); end

% compute svd in transform domain (one-time cost)
AHat             = modeProduct(A,Q');
[UHat,SHat,VHat] = facewiseSVD(AHat);

results = [];
for k = 1:min(size(A,1:2))
    % approximation per frontal slice
    AkHat = facewise(UHat(:,1:k,:),facewise(SHat(1:k,1:k,:), tran(VHat(:,1:k,:))));

    for p = 1:size(A,3)

        % form approximation (only first p frontal slices)
        Ak  = modeProduct(AkHat(:,:,1:p),Q(:,1:p));
    
        % compute error and storage
        err  = fronorm(A - Ak);

        comp = n1 * k * p + n2 * k * p + n3 * p;

        % store results
        results = cat(1,results,[k,p,err,err / nrmA, comp,storeA / comp]);

        % print progress
        fprintf('p = %0.3d, k = %0.3d\n',p,k)
    end
end

% save results for fixed Q as csv file
T = array2table(results, 'VariableNames',headers);
writetable(T,[saveDir,'tsvd/',fileName,'.csv'])


%% projected tensor svdII

if ~exist([saveDir,'tsvdII/'], 'dir'), mkdir([saveDir,'tsvdII/']); end
if ~exist([saveDir,'tsvdII/img'], 'dir'), mkdir([saveDir,'tsvdII/img']); end

kmax    = min(size(A,1:2));
results = zeros(kmax,size(A,3));
storage = zeros(size(results));
gammas  = zeros(size(results));


% compute svd in transform domain (one-time cost)
AHat             = modeProduct(A,Q');
[UHat,SHat,VHat] = facewiseSVD(AHat);

for k = 1:kmax
    disp(['Starting k = ', num2str(k),'...'])

    % form approximation
    AApproxHat = facewise(UHat(:,1:k,:),facewise(SHat(1:k,1:k,:),tran(VHat(:,1:k,:))));

    for p = 1:size(A,3)

        % get energy
        gamma      = fronorm(AApproxHat(:,:,1:p))^2 / fronorm(modeProduct(A,Q(:,1:p)'))^2;
        
        % re-compute projected tsvdII to avoid errors
        [u,s,v,info] = projsvdII(A,gamma,Q(:,1:p));
        
        % compute approximation
        AApprox = projprod(u,projprod(s,tran(v),Q(:,1:p)),Q(:,1:p));
        
        % store results
        results(k,p) = fronorm(A - AApprox);
        storage(k,p) = sum(info.rho(1:p)) * (size(A,1) + size(A,2)) + size(A,3) * p;
        gammas(k,p)  = gamma;
    end
    disp(['Finished p = ', num2str(p),'.'])
end

varNames1 = cellfun(@(x) ['p',num2str(x),'_err'],num2cell(1:size(A,3)),'UniformOutput',false);
varNames2 = cellfun(@(x) ['p',num2str(x),'_store'],num2cell(1:size(A,3)),'UniformOutput',false);
varNames3 = cellfun(@(x) ['gamma',num2str(x)],num2cell(1:size(A,3)),'UniformOutput',false);

varNames = cat(2,{'k'},varNames1,varNames2,varNames3);

results  = [(1:kmax)',results / fronorm(A), numel(A) ./ storage,gammas];
T        = array2table(results,'VariableNames',varNames);

writetable(T,[saveDir,'tsvdII/',fileName,'.csv'])

%% hosvd

if ~exist([saveDir,'hosvd/'], 'dir'), mkdir([saveDir,'hosvd/']); end
if ~exist([saveDir,'hosvd/img'], 'dir'), mkdir([saveDir,'hosvd/img']); end

kRange  = [1,10:10:140,size(A,1)];
results = zeros(length(kRange),length(kRange),size(A,3));
storage = zeros(size(results));

for k3 = 1:size(A,3)
    disp(['Starting k3 = ', num2str(k3),'...'])

    % full HOSVD
    [~,U] = hosvd3(A,size(A));
    
    count1 = 1;
    for k1 = kRange

        count2 = 1;
        for k2 = kRange
            % form core
            G = hosvdCore3(A,U,[k1,k2,k3]);
            
            % form approximation
            HApprox = hosvdApprox3(G,U,[k1,k2,k3]);
            
            % store results
            results(count1,count2,k3) = fronorm(A - HApprox);

            % storage costs
            storage(count1,count2,k3) = k1 * k2 * k3 + k1 * size(A,1) + k2 * size(A,2) + k3 * size(A,3);

            % update counter
            count2 = count2 + 1;
        end
        % update counter
        count1 = count1 + 1;
    end

    disp(['Finished k3 = ', num2str(k3),'.'])
end

varNames1 = cellfun(@(x) ['p',num2str(x),'_err'],num2cell(1:size(A,3)),'UniformOutput',false);
varNames2 = cellfun(@(x) ['p',num2str(x),'_store'],num2cell(1:size(A,3)),'UniformOutput',false);
varNames = cat(2,{'k1','k2'},varNames1,varNames2);

kk1 = kron(ones(1,length(kRange)),kRange);
kk2 = kron(kRange,ones(1,length(kRange)));

results = reshape(results / fronorm(A),[],size(results,3));
storage = reshape(numel(A) ./ storage,[],size(storage,3));
results = [kk1(:),kk2(:),results, storage];

T       = array2table(results,'VariableNames',varNames);

writetable(T,[saveDir,'hosvd/',fileName,'.csv'])


return;

%% setup storage

frame   = 75;
[R,G,B] = deal(26,16,8);

figure(1); clf;
imagesc(rescale(A(:,:,[R,G,B])));
cax = clim;
axis('off'); 
pbaspect([size(A,2) size(A,1) 1])
exportgraphics(gcf,[saveDir,'/img_RGB_.png'])

%% store images for tsvd
results = [];
for p = [20,100]
    Qp      = Q(:,1:p);
    [U,S,V] = projsvd(A,Qp);

    for k = [5,20]
        AApprox = projprod(U(:,1:k,:),projprod(S(1:k,1:k,:),tran(V(:,1:k,:)),Qp),Qp);

        figure(1); clf;
        imagesc(rescale(AApprox(:,:,[R,G,B])));
        pbaspect([size(A,2) size(A,1) 1])
        axis('image'); axis('off');
        exportgraphics(gcf,[saveDir,'tsvd/img/img_RGB_p_',num2str(p),'_k_',num2str(k),'.png']);
       
        rel_err   = fronorm(A - AApprox) / fronorm(A);
        rel_store = numel(A) / (sum(info.rho) * (size(A,1) + size(A,2)) + size(A,3) * p);

        results = cat(1,results,[p,k,rel_err,rel_store]);
    end
end
T = array2table(results,'VariableNames',{'p','k','rel_err','rel_store'});
writetable(T,[saveDir,'tsvd/',fileName,'_small.csv'])


%% store images for tsvdII
results = [];
for p = [20,100]
    Qp = Q(:,1:p);

    for k = [5,20]

        [U,S,V] = projsvd(A,Qp);
        
        AApprox = projprod(U(:,1:k,:),projprod(S(1:k,1:k,:),tran(V(:,1:k,:)),Qp),Qp);
    
        % get energy
        % gamma = fronorm(AApprox)^2 / fronorm(A)^2;
        gamma = fronorm(modeProduct(AApprox,Qp'))^2 / fronorm(modeProduct(A,Qp'))^2;

        
        % compute projected tsvdII
        [u,s,v,info] = projsvdII(A,gamma,Qp);
        
        % recompute approximation
        AApprox      = projprod(u,projprod(s,tran(v),Q),Q);

        figure(1); clf;
        imagesc(rescale(AApprox(:,:,[R,G,B])));
        pbaspect([size(A,2) size(A,1) 1])
        axis('image'); axis('off');
        exportgraphics(gcf,[saveDir,'tsvdII/img/img_RGB_p_',num2str(p),'_k_',num2str(k),'.png']);
       
        rel_err   = fronorm(A - AApprox) / fronorm(A);
        rel_store = numel(A) / (sum(info.rho) * (size(A,1) + size(A,2)) + size(A,3) * p);

        results = cat(1,results,[p,k,gamma,rel_err,rel_store]);
    end
end

T = array2table(results,'VariableNames',{'p','k','gamma','rel_err','rel_store'});
writetable(T,[saveDir,'tsvdII/',fileName,'_small.csv'])

%% images for hosvd

paramList = {[2,5,n1,1],[2,5,9,9],[2,20,n1,1],[2,20,32,32],[100,5,n1,8],[100,5,36,36],[100,20,n1,38],[100,20,74,74]};

[~,U] = hosvd3(A,size(A));

results = [];
for count = 1:length(paramList)
    p = paramList{count}(1);
    k = paramList{count}(2);
    k1 = paramList{count}(3);
    k2 = paramList{count}(4);

    HCore       = hosvdCore3(A,U,[k1,k2,p]);
    HApprox     = hosvdApprox3(HCore,U,[k1,k2,p]);

    figure(1); clf;
    imagesc(rescale(HApprox(:,:,[R,G,B])));
    axis('image'); axis('off');
    exportgraphics(gcf,[saveDir,'hosvd/img/img_RGB_p_',num2str(p),'_k1_',num2str(k1),'_k2_',num2str(k2),'.png']);
    

    rel_err = fronorm(A - HApprox) / fronorm(A);
    rel_store = numel(A) / (k1 * k2 * p + k1 * n1 + k2 * n2 + p * n3);

    results = cat(1,results,[p,k1,k2,rel_err,rel_store]);
end

T = array2table(results,'VariableNames',{'p','k1','k2','rel_err','rel_store'});
writetable(T,[saveDir,'hosvd/',fileName,'_small.csv'])

