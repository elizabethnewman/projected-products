function[d] = tdiag(A,k)
    if ~exist('k','var') || isempty(k), k = 0; end

    if size(A,1) == 1 || size(A,2) == 1
        % outputs tensor with square frontal slices
        % input is numel(idx) x 1 x tubesize or 1 x numel(idx) x tubesize

        % output the diagonal tubes of the tensor
        n   = max(size(A,1),size(A,2));
        szA = size(A);
        nd  = prod(szA(3:end));


        % main diagonal
        idx0 = (0:(n - 1)) * n + (1:n);
    
        % change which diagonal
        idx = idx0 - k;
        idx = idx(idx >= 1 & idx <= n^2);

        % adjust for multiple frontal slices
        idx2 = (0:(nd - 1)) * n^2;

        % form main index
        IDX = kron(ones(1,nd),idx) + kron(idx2,ones(1,numel(idx)));

        % output tensor

        % form tensor
        d = tensor(zeros([n,n,szA(3:end)]));
        s = struct('type','()','subs',{{IDX}});
        d = subsasgn(d,s,A(:)');
        % d(IDX) = A(:)';
    else
        % output the diagonal tubes of the tensor
        m  = size(A,1);
        n  = size(A,2);
        szA = size(A);
        nd = prod(szA(3:end));
        
        r = min(m,n);
    
        % main diagonal
        idx0 = (0:(r - 1)) * m + (1:r);
    
        % change which diagonal
        idx = idx0 - k;
        idx = idx(idx >= 1 & idx <= m * n);
        
        % adjust for multiple frontal slices
        idx2 = (0:(nd - 1)) * (m * n);
    
        % form main index
        IDX = kron(ones(1,nd),idx) + kron(idx2,ones(1,numel(idx)));
    
        % output diag
        % d = reshape(A(IDX),[numel(idx),1,size(A,3:ndims(A))]);
        szA = size(A);
        s = struct('type','()','subs',{{IDX}});
        d = reshape(subsref(A,s),[numel(idx),1,szA(3:end)]);
    end
end

