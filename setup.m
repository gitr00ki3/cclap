addpath(genpath(pwd));
Ddir='./data/';
fname={'NLDRPuncturedSphere','NLDRSineHyperboloid',...
    'NLDRSwissRollwHole','NLDRTwinPeakwHole',...
    'usps','Hasy2','BciHaLT_A','LegoBricks'};
for i=1:length(fname)
    tmp=cell2mat(fname(i));
    if ~exist([tmp '.mat'],'file')
        disp(['Downloading ' tmp '.mat']);
        try
            filestr=websave([Ddir tmp '.mat'],...
                ['https://github.com/gitr00ki3/DivLap/raw/master/data/' tmp '.mat']);
        catch ME
            if strcmp(ME.identifier,'MATLAB:webservices:HTTP404StatusCodeError')
                filestr=websave([Ddir tmp '.mat'],...
                ['https://github.com/gitr00ki3/LapR/raw/master/data/' tmp '.mat']);
            end
        end
        fprintf('%s File saved to %s\n',tmp,filestr);
    end
end

if ~exist('mexGramSVMTrain.mexa64','file');
    disp('Compiling LapSVM...');
    cd('classifier/');
    mex('mexGramSVMTrain.cpp','svmprecomputed.cpp');
    cd('..');
    disp('done...');
end

disp('Setup Completed...');
disp('Run');
disp('nldr_interactive.m- for Non-Linear Dimensionality Reduction examples.');
disp('divRun_v2.m- for manifold regularization classifier examples.');