addpath(genpath(pwd));
%% CODE TO RUN LapSVM AND LapRLSC EXPERIMENTS
if ~exist('elapnorm','var')
    close all;clear;clc;
    dpath='';
    fname={'usps','Hasy2','mnist',...
        'BciHaLT_A',...
        'cifar01','LegoBricks',...
        'UCMercedLand','CVPRv2','NaturalImages'};
    gammaA=[0.05,1e-03,0.181,...
        0.45,...
        0.05,1e-03,...
        0.016,0.073,0.16];
    gammaI=[0.005,0.039,1e-03,...
        1e-03,...
        5e-03,1e-03,...
        0.034,1e-03,0.082];
    NN=[6,5,3,...
        3,...
        8,4,...
        9,3,3];
    methods={'both'};
    weight={'binary','hsic','hsicT'};
    elapnorm=struct();
    i=1;j=1;k=1;
    rfile='results/result_v8.mat';
end
%%
for i=i:length(fname)
    elapnorm(i).fname=cell2mat(fname(i));
    experiment='experiment_all';
    for j=j:length(methods)
        for k=k:length(weight)
            t=1;
            [~, ~, eboth] = feval(experiment,...
                cell2mat(methods(j)),gammaA(i),...
                gammaI(i), NN(i), cell2mat(weight(k)), t,...
                [dpath,cell2mat(fname(i))]);
            
            elapnorm(i).methods(j).method='lapsvm';
            elapnorm(i).methods(j).weights(k).weight=...
                cell2mat(weight(k));
            elapnorm(i).methods(j).weights(k).et=eboth.Set;
            elapnorm(i).methods(j).weights(k).eu=eboth.Seu;
            
            elapnorm(i).methods(j+1).method='laprlsc';
            elapnorm(i).methods(j+1).weights(k).weight=...
                cell2mat(weight(k));
            elapnorm(i).methods(j+1).weights(k).et=eboth.Ret;
            elapnorm(i).methods(j+1).weights(k).eu=eboth.Reu;
        end
        k=1;
    end
    j=1;
    if ~exist(rfile,'file')
        save(rfile,'elapnorm');
    else
        save(rfile,'elapnorm','-append');
    end
end