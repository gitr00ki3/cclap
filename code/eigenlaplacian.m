function L = eigenlaplacian(DATA, options)
% Calculate the graph laplacian of the adjacency graph of data set DATA.
% L = eigenlaplacian(DATA, options)
% DATA - Nxd matrix. Data points are rows.
% options - Data structure containing the following fields
%   1. NN - number of nearest neighbors%
%   2. WEIGHTTYPPE='heat'
%   3. WEIGHTPARAM= width for heat kernel
%   4. NORMALIZE= 0 | 1 whether to return normalized graph laplacian or not
% Returns: L, sparse symmetric NxN matrix

disp('Computing Graph Laplacian.');

NN=options.NN;
WEIGHTTYPE=options.GraphWeights;
WEIGHTPARAM=options.GraphWeightParam;
NORMALIZE=options.GraphNormalize;
disp('Computing weight matrix');
[Idx, A_v]=knnsearch(DATA,DATA,'K',NN+1);
A_v = A_v(:,2:end)';
Idx = Idx(:,2:end)';
Ad_i=(1:size(Idx,2))';
Ad_i=repmat(Ad_i,1,NN)';
Ad_i=Ad_i(:);
Ad_j=Idx(:);
Ad_v=A_v(:);
A=sparse(Ad_i,Ad_j,Ad_v);
[row_a, col_a]=size(A);
if row_a~=col_a
    A = adjacency(DATA, 'nn', NN, 'euclidean');
end
W = A;

% disassemble the sparse matrix
[A_i, A_j, A_v] = find(A);

switch WEIGHTTYPE
    case 'heat'
        disp(['Laplacian : Using Heat Kernel sigma : ' num2str(WEIGHTPARAM)]);
        t=WEIGHTPARAM;
        for i = 1: size(A_i)
            W(A_i(i), A_j(i)) = exp(-A_v(i)^2/(2*t*t));
        end
    case 'binary'
        disp('Laplacian : Using Binary weights ');
        for i = 1: size(A_i)
            W(A_i(i), A_j(i)) = 1;
        end
    case 'hsic'
        disp('Laplacian : Using HSIC');
        [~,n] = size(Idx);
        W = zeros(n);
        for i = 1:n
            Ii = [i; Idx(:,i)];
            Ii = Ii(Ii ~= 0);
            kt = numel(Ii);
            Xi = DATA(Ii,:) - repmat(DATA(i,:), [kt 1]);
            V = abs(hsic(Xi'));
            V = min(V,V');
            W(Ii,Ii) = W(Ii,Ii)+V;
        end
        NORMALIZE=0;
    case 'hsicT'
        disp('Laplacian : Using HSIC_T');
        [~,n] = size(Idx);
        W = zeros(n);
        for i = 1:n
            Ii = [i; Idx(:,i)];
            Ii = Ii(Ii ~= 0);
            kt = numel(Ii);
            Xi = DATA(Ii,:) - repmat(DATA(i,:), [kt 1]);
            tmp = Idx(:,i);
            for j =1:length(tmp)
                Ij = [tmp(j); Idx(:,tmp(j))];
                Ij = Ij(Ij ~= 0);
                kt = numel(Ij);
                Xj = DATA(Ij,:) - repmat(DATA(tmp(j),:), [kt 1]);
                V = abs(hsicT(Xi',Xj'));
%                 V = abs(hsicTT(Xi',Xj',options));
                V = min(V,V');
                W(Ii,Ij) = W(Ii,Ij)+V;
            end
        end
        NORMALIZE=0;
    otherwise
        error('Unknown Weighttype');
end

W = (W+W')/2;
D = sum(W(:,:),2);

if NORMALIZE==0
    L = spdiags(D,0,speye(size(W,1)))-W;
else % normalized laplacian
    D=spdiags(sqrt(1./D),0,length(D),length(D));
    L=eye(size(W,1),'like',sparse(size(W,1),size(W,1),1))-D*W*D;
end

function W = hsic(x)
    [~,n] = size(x);
    H = eye(n) - 1/n*ones(n);
    medx = compmedDist(x');
    Kx = calckernel('rbf',medx,x');
    W = H*Kx*H;
    
function W = hsicT(x,y)
    [~,n] = size(x);
    H = eye(n) - 1/n*ones(n);
    medx = compmedDist(x');
    Kx = calckernel('rbf',medx,x');
    medy = compmedDist(y');
    Ky = calckernel('rbf',medy,y');
    W = Kx*H*Ky*H;