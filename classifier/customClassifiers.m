%% SVM and kNN classifier
function [err_svm, err_knn, err_dtree]=customClassifiers(mapX,train,test,y,yt,NN)
labels = unique(y);
err_svm = zeros(1,length(labels));
err_knn = zeros(1,length(labels));
err_dtree = zeros(1,length(labels));
mapXl = mapX(train, :);
mapXt = mapX(test, :);
MAX=length(labels);
for i =1:MAX
    indx = y==labels(i);
    ytmp = y;
    ytmp(indx) = 1;
    ytmp(~indx) = -1;
    mdl_svm = fitcsvm(mapXl,ytmp,'KernelFunction','rbf',...
        'Standardize',true);
    fprintf('%d/%d SVM Done... ',i,MAX);
    mdl_knn = fitcknn(mapXl,ytmp,'NumNeighbors',NN,...
        'Standardize',1);
    fprintf('%d/%d kNN Done... ',i,MAX);
    mdl_dtree = fitctree(mapXl,ytmp);...
%         ,'PredictorSelection','interaction-curvature',...
%         'OptimizeHyperparameters','auto',...
%         'HyperparameterOptimizationOptions',...
%         struct('Verbose',0,'ShowPlots',false));
    fprintf('%d/%d DT Done...\n',i,MAX);
    
    yt_svm = predict(mdl_svm,mapXt);
    yt_knn = predict(mdl_knn,mapXt);
    yt_dtree = predict(mdl_dtree,mapXt);
    indx = yt==labels(i);
    ytmp = yt;
    ytmp(indx) = 1;
    ytmp(~indx) = -1;
    
    err_svm(1,i)=(sum(yt_svm~=ytmp)/length(yt)*100);
    err_knn(1,i)=(sum(yt_knn~=ytmp)/length(yt)*100);
    err_dtree(1,i)=(sum(yt_dtree~=ytmp)/length(yt)*100);
end