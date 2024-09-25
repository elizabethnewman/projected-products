
clear; clc;

%% load data

video = read(VideoReader('shuttle.avi'));
A = zeros(size(video,1),size(video,2),size(video,4));
for i = 1:size(video,4)
    A(:,:,i) = squeeze(im2gray(video(:,:,:,i)));
end

% normalize
A = A / fronorm(A);

if ~exist('./results_shuttle', 'dir'), mkdir('results_shuttle'); end
if ~exist('./results_shuttle/orig', 'dir'), mkdir('results_shuttle/orig'); end

% choose frames to store
frames = [1,10:10:120];
for i = frames
    imwrite(rescale(A(:,:,i)),sprintf('results_shuttle/orig/img_%0.3d.png',i))
end

[n1,n2,n3]  = size(A);
nrmA        = fronorm(A);
storeA      = n1 * n2 * n3;

[Z,~,~] = svd(modeUnfold(A,3),'econ');

%% matrix approximation

if ~exist(['./results_shuttle/','matrix'], 'dir'), mkdir(['./results_shuttle/','matrix']); end
if ~exist('./results_shuttle/matrix/img', 'dir'), mkdir('./results_shuttle/matrix/img'); end

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

    for j = frames
        imwrite(rescale(AMatk((j-1)*size(A,1)+1:j*size(A,1),:)),sprintf('results_shuttle/%s/img/img_%0.3d_k%0.3d.png','matrix',j,k))
    end

    fprintf('matrix: k = %0.3d\n',k)

end

T = array2table(results, 'VariableNames',headers);
writetable(T,'results_shuttle/matrix/shuttle.csv')

%% form matrices

rng(42);
Q = {'I', eye(n3)', 'C', dctmtx(n3)', 'Z', Z, 'W', orth(randn(n3))};

headers = {'k','p','err','rel_err','comp_ratio','rel_comp_ratio'};

for i = 1:2:length(Q)
    
    if ~exist(['./results_shuttle/',Q{i}], 'dir'), mkdir(['./results_shuttle/',Q{i}]); end
    if ~exist(['./results_shuttle/',Q{i},'/img'], 'dir'), mkdir(['./results_shuttle/',Q{i},'/img']); end
    
    % compute svd in transform domain (one-time cost)
    AHat             = modeProduct(A,Q{i+1}');
    [UHat,SHat,VHat] = facewiseSVD(AHat);

    results = [];
    for k = 1:2
        % approximation per frontal slice
        AkHat = facewise(UHat(:,1:k,:),facewise(SHat(1:k,1:k,:), tran(VHat(:,1:k,:))));

        for p = 1:10

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

            % save figure
            for j = frames
                imwrite(rescale(Ak(:,:,j)),sprintf('results_shuttle/%s/img/img_%0.3d_p%0.3d_k%0.3d.png',Q{i},j,p,k))
            end

            % print progress
            fprintf('Q = %s: p = %0.3d, k = %0.3d\n',Q{i},p,k)
        end
    end
    
    % save results for fixed Q as csv file
    T = array2table(results, 'VariableNames',headers);
    writetable(T,['results_shuttle/',Q{i},'/traffic.csv'])

end





