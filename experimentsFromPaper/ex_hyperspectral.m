
clear; clc;

%% load data

hcube = hypercube('indian_pines.dat');
A     = hcube.DataCube;

% normalize
A = A / fronorm(A);

saveDir  = './results_hyperspectral/';
fileName = 'results';
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
if ~exist([saveDir,'orig'], 'dir'), mkdir([saveDir,'orig']); end

% get sizes
[n1,n2,n3]  = size(A);
nrmA        = fronorm(A);
storeA      = n1 * n2 * n3;

%% form data-dependent transformation
[Z,S,~] = svd(modeUnfold(A,3),'econ');

% get cumulative energy
E = cumsum(diag(S).^2) / sum(diag(S).^2);

% store mode-3 unfolding singular values
T = array2table([(1:size(S,1))',diag(S),E(:)],'VariableNames',{'index', 'sigma', 'cummulative_energy'});
writetable(T,[saveDir,'mode3_singular_values.csv'])

%% matrix approximation

if ~exist([saveDir,'matrix'], 'dir'), mkdir([saveDir,'matrix']); end
if ~exist([saveDir,'matrix/img'], 'dir'), mkdir([saveDir,'matrix/img']); end

AMat = reshape(permute(A,[2,1,3]),size(A,2),[])';

headers = {'k','err','rel_err','comp_ratio','rel_comp_ratio'};

results = [];
[u,s,v] = svd(AMat,0);
for k = 1:size(AMat,2)
    
    % compute matrix SVD
    AMatk = u(:,1:k) * s(1:k,1:k) * v(:,1:k)';

    % compute error and storage
    err  = fronorm(AMat - AMatk);
    comp = n1 * n3 * k + n2 * k;
    
    % store results
    results = cat(1,results,[k,err,err / nrmA, comp,storeA / comp]);

    fprintf('matrix: k = %0.3d\n',k)

end

T = array2table(results, 'VariableNames',headers);
writetable(T,[saveDir,'matrix/',fileName,'.csv'])

%% form matrices

rng(42);
Q = {'I', eye(n3)', 'C', dctmtx(n3)', 'Z', Z, 'W', orth(randn(n3))};

headers = {'k','p','err','rel_err','comp_ratio','rel_comp_ratio'};

for i = 1:2:length(Q)
    
    if ~exist([saveDir,Q{i}], 'dir'), mkdir([saveDir,Q{i}]); end
    if ~exist([saveDir,Q{i},'/img'], 'dir'), mkdir([saveDir,Q{i},'/img']); end
    
    % compute svd in transform domain (one-time cost)
    AHat             = modeProduct(A,Q{i+1}');
    [UHat,SHat,VHat] = facewiseSVD(AHat);

    results = [];
    for k = 1:min(size(A,1:2))
        % approximation per frontal slice
        AkHat = facewise(UHat(:,1:k,:),facewise(SHat(1:k,1:k,:), tran(VHat(:,1:k,:))));

        for p = 1:size(A,3)

            % form approximation (only first p frontal slices)
            Ak  = modeProduct(AkHat(:,:,1:p),Q{i+1}(:,1:p));
        
            % compute error and storage
            err  = fronorm(A - Ak);

            comp = n1 * k * p + n2 * k * p;
            if ~strcmp(Q{i},'I')
                comp = comp + n3 * p;
            end
            
            % store results
            results = cat(1,results,[k,p,err,err / nrmA, comp,storeA / comp]);

            % print progress
            fprintf('Q = %s: p = %0.3d, k = %0.3d\n',Q{i},p,k)
        end
    end
    
    % save results for fixed Q as csv file
    T = array2table(results, 'VariableNames',headers);
    writetable(T,[saveDir,Q{i},'/',fileName,'.csv'])

end

return;

%% create images 

% need to have tensor pre-loaded
K = min(size(A,1:2));
P = size(A,3);

REScales = zeros(2,4);
count = 1;
for name = {'Z', 'I', 'W', 'C'}
    T  = readtable([saveDir,'/',name{1},'/',fileName,'.csv']);
    T  = table2array(T);
    RE = reshape(T(:,4),size(A,3),[]);

    figure(1); clf; 
    imagesc(RE); 
    colormap hot;  
    axis square
    axis off;
    pbaspect([K P 1])
    exportgraphics(gcf,[saveDir,'/',name{1},'/RE.png'])
    
    % store color bounds
    REScales(1,count) = min(RE(:));
    REScales(2,count) = max(RE(:));
    count = count + 1;
end

T = array2table(REScales, 'VariableNames',{'Z', 'I', 'W', 'C'},'RowNames',{'min','max'});
writetable(T,[saveDir,'rel_error_color_scales.csv'], 'writevariablenames', 1)

%% store images

kpPairs = [5, 2; 5, 220];
frame   = 75;
[R,G,B] = deal(26,16,8);

figure(1); clf;
imagesc(rescale(A(:,:,[R,G,B])));
cax = clim;
axis('off'); 
pbaspect([size(A,2) size(A,1) 1])
exportgraphics(gcf,[saveDir,'/img_RGB_.png'])

T = [];
for j = 5
    % transform data
    AHat = modeProduct(A,Q{j + 1}');
    
    % store frobenius norm of transformed frontal slices
    t = vecnorm(reshape(AHat,[],size(A,3)),2,1);
    T = cat(2,T,t(:));
    
    % store approximations
    for i = 1:size(kpPairs,1)
        kk      = kpPairs(i,1);
        pp      = kpPairs(i,2);
        Qp      = Q{j + 1}(:,1:pp);
        [U,S,V] = projsvd(A,Qp,kk);
        Ak      = projprod(U,projprod(S,tran(V),Qp),Qp);
        
        disp(fronorm(A - Ak))

        figure(1); clf;
        imagesc(rescale(Ak(:,:,[R,G,B])));
        axis('off'); 
        pbaspect([size(A,2) size(A,1) 1])
        exportgraphics(gcf,[saveDir,'/',Q{j},'/img/img_RGB_k',num2str(kk),'_p',num2str(pp),'.png'])
    end
end

T = [(1:n3)',T];
T = array2table(T,'VariableNames',{'slice', 'I', 'C', 'Z', 'W'});
writetable(T,[saveDir,'/transform_frobenius_norm.csv'])

