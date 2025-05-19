
clear; clc;

% create saving conventions
saveDir  = './results_helmholtz/';
fileName = 'results';
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
if ~exist([saveDir,'orig'], 'dir'), mkdir([saveDir,'orig']); end


%% solve Helmholtz problem following Matlab tutorial

% number of time points
m = 1500;

% create data
speedRange = [4:9,10:10:100];
A = [];
for ii = speedRange
    % create model
    numberOfPDE = 1;
    model       = createpde(numberOfPDE);

    % PDE parameters
    g = @scatterg;
    c = 1;
    a = -ii^2;
    f = 0;

    % create geometry
    geometryFromEdges(model,g);

    % apply boundary conditions
    bOuter = applyBoundaryCondition(model,"neumann","Edge",(5:8), ...
                                      "g",0,"q",-60i);
    innerBCFunc = @(loc,state)-exp(-1i*ii*loc.x);
    bInner = applyBoundaryCondition(model,"dirichlet","Edge",(1:4), ...
                                          "u",innerBCFunc);
    
    % specify the coefficients
    specifyCoefficients(model,"m",0,"d",0,"c",c,"a",a,"f",f);

    % generate a mesh
    generateMesh(model,"Hmax",0.05);

    % solve PDE
    result = solvepde(model);
    u = result.NodalSolution;

    % create time-dependent solution
    Ak = [];
    for j = 1:m
        uu = real(exp(-j*2*pi/m*sqrt(-1))*u);
        Ak = cat(2,Ak,uu(:));
    end

    A = cat(3,A,Ak);

end


% get size
[n1,n2,n3] = size(A);

% get sizes
storeA = n1 * n2 * n3;
nrmA   = fronorm(A);



% form transformation matrices
rng(42);
[Z,~,~] = svd(modeUnfold(A,3),'econ');
QCell   = {'I', eye(n3)', 'C', dctmtx(n3)', 'Z', Z, 'W', orth(randn(n3)), 'H', generate_haar(n3)'};


%%
% % visualize tensor
% clear M
% 
% for ii = 1:n3
%     k = speedRange(ii);
% 
%     if ~exist([saveDir,'orig/','k',num2str(ii)], 'dir'), mkdir([saveDir,'orig/','k',num2str(ii)]); end
%     for j = [1,101:100:1001]
%         pdeplot(model,"XYData",A(:,j,ii),"ColorBar","off","Mesh","off");
%         colormap(jet)
%         clim([-abs(max(A(:,j,ii))) abs(max(A(:,j,ii)))]);
%         axis tight
%         ax = gca;
%         ax.DataAspectRatio = [1 1 1]; 
%         axis off
%         exportgraphics(gcf,[saveDir,'orig/','k',num2str(ii),'/frame',num2str(j),'.png'])
%         % M(j) = getframe;
%     end
% end


%% compute projected tensor SVD

% maximum rank of frontal slices
% kmax    = 2;

% % run experiment
% headers = {'k','p','err','rel_err','comp_ratio','rel_comp_ratio'};
% 
% if ~exist([saveDir,'tsvd/'], 'dir'), mkdir([saveDir,'tsvd/']); end
% 
% 
% % compute svd in transform domain (one-time cost)
% for q = 1:2:length(QCell)
% 
%     % choose transformation matrix
%     QName   = QCell{q};
%     Q       = QCell{q + 1};
% 
%     fprintf('--------------------------------------------------\n')
%     fprintf('%s\n',QName)
%     fprintf('--------------------------------------------------\n')
% 
%     if ~exist([saveDir,'tsvd/',QName], 'dir'), mkdir([saveDir,'tsvd/',QName]); end
%     if ~exist([saveDir,'tsvd/',QName,'/img'], 'dir'), mkdir([saveDir,'tsvd/',QName,'/img']); end
% 
%     % move to transform domain
%     AHat             = modeProduct(A,Q');
%     [UHat,SHat,VHat] = facewiseSVD(AHat);
% 
%     results = [];
%     for k = 1:kmax
%         % approximation per frontal slice
%         AkHat = facewise(UHat(:,1:k,:),facewise(SHat(1:k,1:k,:), tran(VHat(:,1:k,:))));
% 
%         for p = 1:size(A,3)
% 
%             % form approximation (only first p frontal slices)
%             Ak  = modeProduct(AkHat(:,:,1:p),Q(:,1:p));
% 
%             % compute error and storage
%             err  = fronorm(A - Ak);
% 
%             comp = n1 * k * p + n2 * k * p + n3 * p;
% 
%             % store results
%             results = cat(1,results,[k,p,err,err / nrmA, comp,storeA / comp]);
% 
%             % print progress
%             fprintf('p = %0.3d\tk = %0.3d\terr = %0.2e\tcomp = %0.2e\n',p,k,err / nrmA,storeA / comp)
%         end
%     end
% 
%     % save results for fixed Q as csv file
%     T = array2table(results, 'VariableNames',headers);
%     writetable(T,[saveDir,'tsvd/',QName,'/',fileName,'.csv'])
% end
% 

%% projected tensor svdII

if ~exist([saveDir,'tsvdII/'], 'dir'), mkdir([saveDir,'tsvdII/']); end

kRange = 1:2;
pRange = 1:size(A,3);

for q = 1:2:length(QCell)

    % choose transformation matrix
    QName   = QCell{q};
    Q       = QCell{q + 1};

    fprintf('--------------------------------------------------\n')
    fprintf('%s\n',QName)
    fprintf('--------------------------------------------------\n')
    
    if ~exist([saveDir,'tsvdII/',QName], 'dir'), mkdir([saveDir,'tsvdII/',QName]); end
    if ~exist([saveDir,'tsvdII/',QName,'/img'], 'dir'), mkdir([saveDir,'tsvdII/',QName,'/img']); end
    
    % reset
    results = zeros(length(kRange),size(A,3));
    storage = zeros(size(results));
    gammas  = zeros(size(results));

    % compute svd in transform domain (one-time cost)
    AHat             = modeProduct(A,Q');
    [UHat,SHat,VHat] = facewiseSVD(AHat);

    for k = kRange

        % compute approximation to obtain energy 
        AApproxHat = facewise(UHat(:,1:k,:),facewise(SHat(1:k,1:k,:),tran(VHat(:,1:k,:))));

        % for p = 1:size(A,3)
        for p = pRange
            disp(['Starting p = ', num2str(p),'...'])
        
            % get energy
            gamma = fronorm(AApproxHat(:,:,1:p))^2 / fronorm(AHat(:,:,1:p))^2;
            
            % re-compute projected tsvdII to avoid errors
            [u,s,v,info] = projsvdII(A,gamma,Q(:,1:p));
            
            % compute approximation
            AApprox = projprod(u,projprod(s,tran(v),Q(:,1:p)),Q(:,1:p));
            
            % store results
            results(k,p) = fronorm(A - AApprox);
            storage(k,p) = sum(info.rho) * (size(A,1) + size(A,2)) + size(A,3) * p;
            gammas(k,p)  = gamma;
        end
        disp(['Finished p = ', num2str(p),'.'])
    end
    
    kk      = kron(kRange(:)',ones(size(A,3),1));
    pp      = kron(ones(length(kRange),1),pRange(:)');
    gammas  = gammas(:)';
    results = results';
    storage = storage';

    % concatenate results
    results  = [kk,pp,gammas(:),results(:),results(:) / fronorm(A), storage(:), numel(A) ./ storage(:)];
    T        = array2table(results,'VariableNames',{'k','p','gamma','err','rel_err','comp_ratio','rel_comp_ratio'});
    
    writetable(T,[saveDir,'tsvdII/',QName,'/',fileName,'.csv'])
end

%% hosvd

if ~exist([saveDir,'hosvd/'], 'dir'), mkdir([saveDir,'hosvd/']); end
if ~exist([saveDir,'hosvd/img'], 'dir'), mkdir([saveDir,'hosvd/img']); end


% kmaxH = max(rank(modeUnfold(A,1)),rank(modeUnfold(A,2)));
kmaxH = 16;


% store spectra
s1 = svd(modeUnfold(A,1),'econ'); m = numel(s1);
s2 = svd(modeUnfold(A,2),'econ'); m = max(m,numel(s2));
s3 = svd(modeUnfold(A,3),'econ'); m = max(m,numel(s3));

T  = array2table([...
    [s1(:); NaN*ones(m-numel(s1),1)],...
    [s2(:); NaN*ones(m-numel(s2),1)],...
    [s3(:); NaN*ones(m-numel(s3),1)]],'VariableNames',{'s1','s2','s3'});
writetable(T,[saveDir,'hosvd/mode_unfolding_spectra.csv'])


kRange  = 1:kmaxH;
results = zeros(length(kRange),length(kRange),size(A,3));
storage = zeros(size(results));


% full HOSVD
[~,U] = hosvd3(A,size(A));

for k3 = pRange
    disp(['Starting k3 = ', num2str(k3),'...'])

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

kk1 = kron(kron(ones(1,length(kRange)),kRange),ones(1,size(A,3)));
kk2 = kron(kron(kRange,ones(1,length(kRange))),ones(1,size(A,3)));
pp  = kron(kron(ones(1,length(kRange)),ones(1,length(kRange))),1:size(A,3));

results = reshape(results,[],size(results,3))';
storage = reshape(storage,[],size(storage,3))';
 
results = [kk1(:),kk2(:),pp(:),results(:), results(:) ./ fronorm(A),storage(:),numel(A) ./ storage(:)];

T       = array2table(results,'VariableNames',{'k1','k2','p','err','rel_err','comp_ratio','rel_comp_ratio'});

writetable(T,[saveDir,'hosvd/',fileName,'.csv'])

% create separate table for each k1 for easy plotting
for k1 = kRange
    idx = (kk1 == k1);
    r1  = results(idx,:);
    T1 = array2table(r1(:,2:end),'VariableNames',{'k2','p','err','rel_err','comp_ratio','rel_comp_ratio'});
    writetable(T1,[saveDir,'hosvd/',fileName,'_k1_',num2str(k1),'.csv'])
end


return;

%% setup storage

imgRange = 1:10:size(A,3);

for ii = imgRange
    imwrite(rescale(A(:,:,ii)),[saveDir,'/orig/img_',num2str(ii),'.png'])
end

%% store images for tsvd
results = [];
for p = [20,100]
    Qp      = Q(:,1:p);
    [U,S,V] = projsvd(A,Qp);

    for ii = [5,20]
        AApprox = projprod(U(:,1:ii,:),projprod(S(1:ii,1:ii,:),tran(V(:,1:ii,:)),Qp),Qp);

        for ii = imgRange
            imwrite(rescale(AApprox(:,:,ii)),[saveDir,'/tsvd/img/img_',num2str(ii),'_p_',num2str(p),'_k_',num2str(ii),'.png'])
        end
        rel_err   = fronorm(A - AApprox) / fronorm(A);
        rel_store = numel(A) / (sum(info.rho) * (size(A,1) + size(A,2)) + size(A,3) * p);

        results = cat(1,results,[p,ii,rel_err,rel_store]);
    end
end
T = array2table(results,'VariableNames',{'p','k','rel_err','rel_store'});
writetable(T,[saveDir,'tsvd/',fileName,'_small.csv'])


%% store images for tsvdII
results = [];
for p = [20,100]
    Qp = Q(:,1:p);

    for ii = [5,20]

        [U,S,V] = projsvd(A,Qp);
        
        AApprox = projprod(U(:,1:ii,:),projprod(S(1:ii,1:ii,:),tran(V(:,1:ii,:)),Qp),Qp);
    
        % get energy
        % gamma = fronorm(AApprox)^2 / fronorm(A)^2;
        gamma = fronorm(modeProduct(AApprox,Qp'))^2 / fronorm(modeProduct(A,Qp'))^2;

        
        % compute projected tsvdII
        [u,s,v,info] = projsvdII(A,gamma,Qp);
        
        % recompute approximation
        AApprox      = projprod(u,projprod(s,tran(v),Q),Q);

        for ii = imgRange
            imwrite(rescale(AApprox(:,:,ii)),[saveDir,'/tsvd/img/img_',num2str(ii),'_p_',num2str(p),'_k_',num2str(ii),'.png'])
        end
        
        rel_err   = fronorm(A - AApprox) / fronorm(A);
        rel_store = numel(A) / (sum(info.rho) * (size(A,1) + size(A,2)) + size(A,3) * p);

        results = cat(1,results,[p,ii,gamma,rel_err,rel_store]);
    end
end

T = array2table(results,'VariableNames',{'p','k','gamma','rel_err','rel_store'});
writetable(T,[saveDir,'tsvdII/',fileName,'_small.csv'])

%% images for hosvd

% paramList = {[2,5,n1,1],[2,5,9,9],[2,20,n1,1],[2,20,32,32],[100,5,n1,8],[100,5,36,36],[100,20,n1,38],[100,20,74,74]};
paramList = {[2,5,n1,1],[2,5,9,9],[2,20,n1,1],[2,20,32,32]};


[~,U] = hosvd3(A,size(A));

results = [];
for count = 1:length(paramList)
    p = paramList{count}(1);
    ii = paramList{count}(2);
    k1 = paramList{count}(3);
    k2 = paramList{count}(4);

    HCore       = hosvdCore3(A,U,[k1,k2,p]);
    HApprox     = hosvdApprox3(HCore,U,[k1,k2,p]);

    for ii = imgRange
        imwrite(rescale(HApprox(:,:,ii)),[saveDir,'/hosvd/img/img_',num2str(ii),'_k1_',num2str(k1),'_k2_',num2str(k2),'.png'])
    end

    rel_err = fronorm(A - HApprox) / fronorm(A);
    rel_store = numel(A) / (k1 * k2 * p + k1 * n1 + k2 * n2 + p * n3);

    results = cat(1,results,[p,k1,k2,rel_err,rel_store]);
end

T = array2table(results,'VariableNames',{'p','k1','k2','rel_err','rel_store'});
writetable(T,[saveDir,'hosvd/',fileName,'_small.csv'])

