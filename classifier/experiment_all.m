function [et,eu,eboth,alphas,bias]=experiment_all(method,...
    gamma_A, gamma_I, NN, G_WEIGHT, GW_PARAM, file)
    if ~exist([file '.mat'],'file')
        fprintf('%s.mat does not exists',file);
        et=inf;eu=inf;
        eboth.Set=inf;eboth.Seu=inf;
        eboth.Ret=inf;eboth.Reu=inf;
        return;
    else
        load(file);
    end
    [MAX,n]=size(train);
    if MAX<n
        MAX=n;
    end
    l=size(Labels, 3)/2;

    options=ml_options('Kernel', kernel, 'KernelParam', kernelParam, ...
        'NN', NN, 'gamma_A', gamma_A, 'gamma_I', gamma_I, ...
        'GraphWeights', G_WEIGHT, 'GraphWeightParam', GW_PARAM);

    p=0;
    for i=1:MAX
        for j=i+1:MAX
            p=p+1;
            x=[train{i};train{j}];
            xt=[test{i};test{j}];
            if isa(x,'uint8')||~isempty(strfind(lower(file),'hasy'))...
                    ||~isempty(strfind(lower(file),'cvpr'))...
                    ||~isempty(strfind(lower(file),'ucmerced'))...
                    ||~isempty(strfind(lower(file),'cifar01'))
                x=zscore(im2double(x));
                xt=zscore(im2double(xt));
            end
            yt=[ones(size(test{i},1),1); -ones(size(test{j},1),1)];
            ytrue=[ones(size(train{i},1),1); -ones(size(train{j},1),1)];
            disp('Computing Kernels');
            K=calckernel(kernel,kernelParam,x);
            KT=calckernel(kernel,kernelParam,x,xt);
            disp('Done.');
            
            if strcmp(method,'lapsvm') || strcmp(method, 'laprlsc')...
                    || strcmp(method, 'both')
                if strcmp(G_WEIGHT,'hessian')
                    TanParam.DimGiven=0;
                    TanParam.EValueTolerance=0.75;
                    L=ConstructHessian(x,GetKNN(x,NN),TanParam);
                else
                    L=feval('eigenlaplacian',x, options);
                end
                L(isnan(L)|isinf(L))=0;
            end

            for k=1:10
                ypos=zeros(size(train{i},1),1);
                ypos(Labels(p,k,1:l))=1;
                yneg=zeros(size(train{j},1),1);
                yneg(Labels(p,k,l+1:2*l))=-1;
                y=[ypos;yneg];
                lab=find(y);
                unlab=find(y==0);
                yu=ytrue(unlab);

                switch method
                    case {'laprlsc','lapsvm'}
                        [alpha,b]=feval(method,K,y,L,options.gamma_A,options.gamma_I);
                        fu=K(unlab,:)*alpha;
                        ft=KT*alpha;
                        alphas{p,k}=alpha;
                        bias(p,k)=b;
                        bt=breakeven(ft,yt,@pre_rec_equal);
                        bu=breakeven(fu,yu,@pre_rec_equal);
                        et(p,k)=evaluate(sign(ft-bt),yt);
                        eu(p,k)=evaluate(sign(fu-bu),yu);
                        disp([p k et(p,k) eu(p,k)]);
                    case 'both'
                        [alpha,b]=feval('lapsvm',K,y,L,options.gamma_A,options.gamma_I);
                        fu=K(unlab,:)*alpha;
                        ft=KT*alpha;
                        eboth.Salphas{p,k}=alpha;
                        eboth.Sbias(p,k)=b;
                        bt=breakeven(ft,yt,@pre_rec_equal);
                        bu=breakeven(fu,yu,@pre_rec_equal);
                        eboth.Set(p,k)=evaluate(sign(ft-bt),yt);
                        eboth.Seu(p,k)=evaluate(sign(fu-bu),yu);
                        fprintf('%s %d %d %.3f %.3f\n',...
                            'lapsvm', p, k, eboth.Set(p,k), eboth.Seu(p,k));
                        
                        [alpha,b]=feval('laprlsc',K,y,L,options.gamma_A,options.gamma_I);
                        fu=K(unlab,:)*alpha;
                        ft=KT*alpha;
                        eboth.Ralphas{p,k}=alpha;
                        eboth.Rbias(p,k)=b;
                        bt=breakeven(ft,yt,@pre_rec_equal);
                        bu=breakeven(fu,yu,@pre_rec_equal);
                        eboth.Ret(p,k)=evaluate(sign(ft-bt),yt);
                        eboth.Reu(p,k)=evaluate(sign(fu-bu),yu);
                        fprintf('%s %d %d %.3f %.3f\n',...
                            'laprlsc', p, k, eboth.Ret(p,k), eboth.Reu(p,k));
                end
            end
        end
    end
    if strcmpi(method,'both')
        et=inf;eu=inf;
    end
end

function e=evaluate(a,b)
    e=sum(a~=b)/length(b)*100;
end