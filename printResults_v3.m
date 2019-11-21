clear;
load result_v8.mat;
clearvars -except elapnorm;
%%
rstwgt=1:3;%[1 5 6 2 3];%BINARY EMR24 EMR72 HSIC_L HSIC_N
stdw = 1;
m=1;
tbs=cell(length(elapnorm),4);
for i=1:length(elapnorm)
    for j=1:length(elapnorm(i).methods)
        meanEt=[]; stdEt=[]; meanEu=[]; stdEu=[];
        for loop=1:length(elapnorm(i).methods(j).weights)
            k=rstwgt(loop);
            meanEt=[meanEt, mean(mean(elapnorm(i).methods(j).weights(k).et,2))];
            stdEt=[stdEt, std(std(elapnorm(i).methods(j).weights(k).et,stdw,2))];
            meanEu=[meanEu, mean(mean(elapnorm(i).methods(j).weights(k).eu,2))];
            stdEu=[stdEu, std(std(elapnorm(i).methods(j).weights(k).eu,stdw,2))];
        end
        [~,minEt]=min(meanEt); [~,minEu]=min(meanEu);
        tabEt=cell(1); tabEu= cell(1);
        for l=1:length(meanEt)
            if minEt==l
                temp=['$ \mathbf{' num2str(meanEt(l)) '} $\\$ \pm' ...
                    num2str(stdEt(l)) ' $'];
            else
                temp=['$ ' num2str(meanEt(l)) ' $\\$ \pm' ...
                    num2str(stdEt(l)) ' $'];
            end
            temp=strrep(temp,' $\\$ ',''); % Comment if new line is req.
            tabEt{1}=[tabEt{1} ' & ' temp];

            if minEu==l
                temp=['$ \mathbf{' num2str(meanEu(l)) '} $\\$ \pm' ...
                    num2str(stdEu(l)) ' $'];
            else
                temp=['$ ' num2str(meanEu(l)) ' $\\$ \pm' ...
                    num2str(stdEu(l)) ' $'];
            end
            temp=strrep(temp,' $\\$ ',''); % Comment if new line is req.
            tabEu{1}=[tabEu{1} ' & ' temp];
        end
        index=((j-1)*j)+1;
        tbs{m,index}=[elapnorm(i).fname ' & ' elapnorm(i).methods(j).method ' & Test'...
            tabEt{1} '\\\hline'];
        tbs{m,index+1}=[elapnorm(i).fname ' & ' elapnorm(i).methods(j).method ' & Unlab'...
            tabEu{1} '\\\hline'];
    end
    m=m+1;
end
% clearvars -except tbs;