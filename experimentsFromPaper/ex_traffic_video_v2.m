
clear; clc;

saveDir  = './results_traffic_v2/';
fileName = 'results';
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
if ~exist([saveDir,'orig'], 'dir'), mkdir([saveDir,'orig']); end



%% create synthetic data

video = read(VideoReader('traffic.mj2'));
A = zeros(size(video,1),size(video,2),size(video,4));
for i = 1:size(video,4)
    A(:,:,i) = squeeze(im2gray(video(:,:,:,i)));
end

% normalize
A = A / fronorm(A);

% permute
A = permute(A,[2,3,1]);


% get sizes
[n1,n2,n3]  = size(A);
nrmA        = fronorm(A);
storeA      = n1 * n2 * n3;

%% create matrices
[Z,~,~] = svd(modeUnfold(A,3),'econ');
QCell   = {'I', eye(n3)', 'C', dctmtx(n3)', 'Z', Z, 'W', orth(randn(n3))};

%% compute projected tensor SVD with Q = Z (repeat of previous experiment)


% run experiment
headers = {'k','p','err','rel_err','comp_ratio','rel_comp_ratio'};

if ~exist([saveDir,'tsvd/'], 'dir'), mkdir([saveDir,'tsvd/']); end


% compute svd in transform domain (one-time cost)
for q = 1:2:length(QCell)

    % choose transformation matrix
    QName   = QCell{q};
    Q       = QCell{q + 1};

    fprintf('--------------------------------------------------\n')
    fprintf('%s\n',QName)
    fprintf('--------------------------------------------------\n')
    
    if ~exist([saveDir,'tsvd/',QName], 'dir'), mkdir([saveDir,'tsvd/',QName]); end
    if ~exist([saveDir,'tsvd/',QName,'/img'], 'dir'), mkdir([saveDir,'tsvd/',QName,'/img']); end

    % move to transform domain
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
            if ~mod(k,10)
                fprintf('p = %0.3d\tk = %0.3d\terr = %0.2e\tcomp = %0.2e\n',p,k,err / nrmA,storeA / comp)
            end
        end
    end
    
    % save results for fixed Q as csv file
    T = array2table(results, 'VariableNames',headers);
    writetable(T,[saveDir,'tsvd/',QName,'/',fileName,'.csv'])
end


%% projected tensor svdII

if ~exist([saveDir,'tsvdII/'], 'dir'), mkdir([saveDir,'tsvdII/']); end

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
    kmax    = min(size(A,1:2));
    results = zeros(kmax,size(A,3));
    storage = zeros(size(results));
    gammas  = zeros(size(results));

    % compute svd in transform domain (one-time cost)
    AHat             = modeProduct(A,Q');
    [UHat,SHat,VHat] = facewiseSVD(AHat);

    for k = 1:kmax

        % compute approximation to obtain energy 
        AApproxHat = facewise(UHat(:,1:k,:),facewise(SHat(1:k,1:k,:),tran(VHat(:,1:k,:))));

        for p = 1:size(A,3)
            disp(['Starting p = ', num2str(p),'...'])
        
            % get energy
            gamma      = fronorm(AApproxHat(:,:,1:p))^2 / fronorm(AHat(:,:,1:p))^2;
            
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
    
    varNames1 = cellfun(@(x) ['p',num2str(x),'_err'],num2cell(1:size(A,3)),'UniformOutput',false);
    varNames2 = cellfun(@(x) ['p',num2str(x),'_store'],num2cell(1:size(A,3)),'UniformOutput',false);
    varNames3 = cellfun(@(x) ['gamma',num2str(x)],num2cell(1:size(A,3)),'UniformOutput',false);
    
    varNames = cat(2,{'k'},varNames1,varNames2,varNames3);
    
    results  = [(1:kmax)',results / fronorm(A), numel(A) ./ storage,gammas];
    T        = array2table(results,'VariableNames',varNames);
    
    writetable(T,[saveDir,'tsvdII/',QName,'/',fileName,'.csv'])
end

%% projected tensor svdII by energy

if ~exist([saveDir,'tsvdII_energy/'], 'dir'), mkdir([saveDir,'tsvdII_energy/']); end

gammaRange = linspace(0.9,1,101);

for q = 1:2:length(QCell)

    % choose transformation matrix
    QName   = QCell{q};
    Q       = QCell{q + 1};

    fprintf('--------------------------------------------------\n')
    fprintf('%s\n',QName)
    fprintf('--------------------------------------------------\n')
    
    if ~exist([saveDir,'tsvdII_energy/',QName], 'dir'), mkdir([saveDir,'tsvdII_energy/',QName]); end
    if ~exist([saveDir,'tsvdII_energy/',QName,'/img'], 'dir'), mkdir([saveDir,'tsvdII_energy/',QName,'/img']); end
    
    % reset
    results = zeros(length(gammaRange),size(A,3));
    storage = zeros(size(results));
    
    for g = 1:length(gammaRange)

        for p = 1:size(A,3)
            disp(['Starting p = ', num2str(p),'...'])
        
            % re-compute projected tsvdII to avoid errors
            [u,s,v,info] = projsvdII(A,gammaRange(g),Q(:,1:p));
            
            % compute approximation
            AApprox = projprod(u,projprod(s,tran(v),Q(:,1:p)),Q(:,1:p));
            
            % store results
            results(g,p) = fronorm(A - AApprox);
            storage(g,p) = sum(info.rho) * (size(A,1) + size(A,2)) + size(A,3) * p;
        end
        disp(['Finished p = ', num2str(p),'.'])
    end
    
    varNames1 = cellfun(@(x) ['p',num2str(x),'_err'],num2cell(1:size(A,3)),'UniformOutput',false);
    varNames2 = cellfun(@(x) ['p',num2str(x),'_store'],num2cell(1:size(A,3)),'UniformOutput',false);    
    varNames = cat(2,{'gamma'},varNames1,varNames2);
    
    results  = [gammaRange(:),results / fronorm(A), numel(A) ./ storage];
    T        = array2table(results,'VariableNames',varNames);
    
    writetable(T,[saveDir,'tsvdII_energy/',QName,'/',fileName,'.csv'])
end

%% hosvd

if ~exist([saveDir,'hosvd/'], 'dir'), mkdir([saveDir,'hosvd/']); end
if ~exist([saveDir,'hosvd/img'], 'dir'), mkdir([saveDir,'hosvd/img']); end

k1Range  = [1,10:10:size(A,1)];
k2Range  = [1,10:10:size(A,2)];
results = zeros(length(k1Range),length(k2Range),size(A,3));
storage = zeros(size(results));

for k3 = 1:size(A,3)
    disp(['Starting k3 = ', num2str(k3),'...'])

    % full HOSVD
    [~,U] = hosvd3(A,size(A));
    
    count1 = 1;
    for k1 = k1Range

        count2 = 1;
        for k2 = k2Range
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

kk1 = kron(ones(1,length(k2Range)),k1Range);
kk2 = kron(k2Range,ones(1,length(k1Range)));

results = reshape(results / fronorm(A),[],size(results,3));
storage = reshape(numel(A) ./ storage,[],size(storage,3));
results = [kk1(:),kk2(:),results, storage];

T       = array2table(results,'VariableNames',varNames);

writetable(T,[saveDir,'hosvd/',fileName,'.csv'])


return;


%% ratios

T1 = readtable('results_traffic_v2/tsvdII/Z/results.csv');
T2 = readtable('results_traffic_v2/hosvd/results.csv');

n3 = 120;
PP = [];
for p = 1:n3
    % find relevant columns of Q-SVDII
    err1  = T1.(['p',num2str(p),'_err']);
    comp1 = T1.(['p',num2str(p),'_store']);

    % what it means to be "better"
    [~,idx] = unique(err1);

    % interpolate
    x  = err1(idx);
    y  = comp1(idx);
    pp = interp1(x,y,'linear','pp');
    xx = linspace(0,1,201);
    yy = ppval(pp,xx);

    loglog(xx,yy)
    hold on;
        loglog(err1,comp1,'o')
    
    % find number of points of HOSVD that is above interpolant
    count = 0;
    for i = 1:n3
        err2  = T2.(['p',num2str(i),'_err']);
        comp2 = T2.(['p',num2str(i),'_store']); 
        yy2   = ppval(pp,err2);
        count = count + sum(comp2 < yy2);
        

    end
    
    PP = cat(1,PP,count);
    % fprintf('p = %d, percent = %0.2f\n',p,100 * count / (n3 * size(T2,1)))
    
end

T = array2table([(1:n3)',PP(:),100 * PP(:) / (n3 * size(T2,1))],'VariableNames',{'p','count','percent'});

writetable(T,[saveDir,'Z_vs_hosvd_percent.csv'])
% figure(1); clf;
% loglog(xx,yy)
% hold on;
% loglog(err1,comp1,'o')
% loglog(err2,comp2,'x')
% hold off;

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

    for k = [5,20]
        AApprox = projprod(U(:,1:k,:),projprod(S(1:k,1:k,:),tran(V(:,1:k,:)),Qp),Qp);

        for ii = imgRange
            imwrite(rescale(AApprox(:,:,ii)),[saveDir,'/tsvd/img/img_',num2str(ii),'_p_',num2str(p),'_k_',num2str(k),'.png'])
        end
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

        for ii = imgRange
            imwrite(rescale(AApprox(:,:,ii)),[saveDir,'/tsvd/img/img_',num2str(ii),'_p_',num2str(p),'_k_',num2str(k),'.png'])
        end
        
        rel_err   = fronorm(A - AApprox) / fronorm(A);
        rel_store = numel(A) / (sum(info.rho) * (size(A,1) + size(A,2)) + size(A,3) * p);

        results = cat(1,results,[p,k,gamma,rel_err,rel_store]);
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
    k = paramList{count}(2);
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

