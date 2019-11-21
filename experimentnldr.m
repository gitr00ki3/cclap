function mapX=experimentnldr(FILE, method, D, nn)
% FILE- Give .mat file name containing
%   1. data- nxd format where n=#of observations and d=ambient dimension
%   2. D- target dimension if explicitly D not passed to function or D=0
% method- method name lap, lap_ad, lle, isomap, ltsa
% nn- #of nearest neighborhood for methods other than lap_ad, for lap_ad
%     provide DELTA value as mentioned in paper.
% t- Parzen window for Laplacian, if not given or t=0, Silverman's rule of thumb
%    will be used.
    load(FILE,'data');
    switch method
        case 'PCA'
            disp('Computing PCA');
            [~,mapX] = pca(data,'NumComponents',D);
        case 'MDS'
            disp('Computing MDS');
            [Idx, A_v]=knnsearch(data,data,'K',nn+1);
            A_v = A_v(:,2:end)';
            Idx = Idx(:,2:end)';
            Ad_i=(1:size(Idx,2))';
            Ad_i=repmat(Ad_i,1,nn)';
            Ad_i=Ad_i(:);
            Ad_j=Idx(:);
            Ad_v=A_v(:);
            A=sparse(Ad_i,Ad_j,Ad_v);
            A=(A'+A)/2;
            opts = statset('Display','iter','MaxIter',10,'TolFun',1e-2);
            mapX = mdscale(full(A),D,'Options',opts);
        case 'HESSIAN'
            TanParam.NCoordDim=2;
            TanParam.DimGiven=1;
            B=ConstructHessian(data,GetKNN(data,nn),TanParam);
            B(isnan(B)|isinf(B))=0;
            [mapX, ~] = eigs(sparse(B),D,'sm');
        case 'LE'
            t = 1.06*mean(std(data))*nthroot(size(data,1),5);
            mapX=graphLaplacian(data,nn,t,D,'heat','eigenlaplacian');
        case 'LLE'
            % Local Linear Embedding
            mapX=lle(data',nn,D)';
        case 'ISOMAP'
            % Isometric Mapping
            options.dims = D;
            options.display = 0;
            options.overlay = 0;
            A=pdist2(data,data);
            mapX = Isomap(A,'k', nn, options);
            mapX=mapX.coords{1,1}';
        case 'LTSA'
            % Local Tangent Space Alignment
            mapX=ltsa(data, D, nn);
        case 'HSIC-LTSA'
            % Local Tangent Space Alignment
            mapX=hsicltsa(data, D, nn);
        case 'HSIC-LE'
            t = 0;
            mapX=graphLaplacian(data,nn,t,D,'hsic','eigenlaplacian');
        case 'HSICT-LE'
            t = 0;
            mapX=graphLaplacian(data,nn,t,D,'hsicT','eigenlaplacian');
        case 'HLE'
            t = compmedDist(data);
            mapX=graphLaplacian(data,nn,t,D,'heat','HminusW');
        otherwise
            error('Valid methods include');
    end
end

function mapX=graphLaplacian(data,NN,t,D,weight,funcname)
    options = ml_options('GraphNormalize',1);
    options.NN=NN;
    options.GraphWeights = weight;
    options.GraphWeightParam = t;
    L = feval(funcname,data, options);
    L(isnan(L)|isinf(L))=0;
    [mapX, ~] = eigs(sparse(L),D+1,'sr');
    mapX = mapX(:,2:end);
end