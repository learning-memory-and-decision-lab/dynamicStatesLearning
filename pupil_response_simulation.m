
% Run the two main codes first (oddball_condition_main.m &
% changepoint_condition_main.m to save the required variables):


load('pCha.mat')
load('pOdd.mat')
load('feedbackOdd.mat')
load('feedbackChange.mat')
load('oddMean.mat')
load('cMean.mat')
load('oddSTD.mat')
load('cSTD.mat')
load('oddUpdate.mat')
load('changeUpdate.mat')

% Simulate Data:
for simSub =1:32
   
    y=gampdf(1:2000,45,30);
    
    pMat=zeros(480*2,2000);
    
    pMat(1:480,400)=oddProbCond(simSub,:)';
    pMat(1:480,900)=oddProbCond(simSub,:)';
    pMat(481:end,400)=changeProbCond(simSub,:)';
    
    
    clear y2
    for i=1:480*2
        y2(i,:)=conv(pMat(i,:),y,'same');
    end
    
    isCPCondition=[ false(480,1); true(480,1);];
    
    surprise=[oddProbCond(simSub,:)'; changeProbCond(simSub,:)'];
  
    D_KL= zeros(960,1);
    I = D_KL;
    for i=1:479
        PD_t = normpdf(1:300,oddMean(simSub,i+1),oddSTD(simSub,i+1));
        PD_t=PD_t./sum(PD_t);
        PD_t_1=normpdf(1:300,oddMean(simSub,i),oddSTD(simSub,i));
        PD_t_1=PD_t_1./sum(PD_t_1);
        D_KL(i+1)=PD_t_1*log10(PD_t_1./PD_t)';
        C_PD_t = normpdf(1:300,changeMean(simSub,i+1),changeSTD(simSub,i+1));
        C_PD_t=C_PD_t./sum(C_PD_t);
        CPD_t_1=normpdf(1:300,changeMean(simSub,i),changeSTD(simSub,i));
        CPD_t_1=CPD_t_1./sum(CPD_t_1);
        I(i)=-log10(normpdf(oddMean(simSub,i+1),oddMean(simSub,i),oddSTD(simSub,i)));
        I(i+480)=-log10(normpdf(changeMean(simSub,i+1),changeMean(simSub,i),changeSTD(simSub,i)));
        D_KL(i+481)=CPD_t_1*log10(CPD_t_1./C_PD_t)';        
                
    end
    
    
   
    %surpriseByCondition=(surprise-mean(surprise)).*(isCPCondition-mean(isCPCondition));
    surpriseByCondition=[oddUpdate(simSub,:)' ;changeUpdate(simSub,:)'];
    
%    [r,p]=corrcoef(surprise(481:end),I(481:end));
   [r,p]=corrcoef(surpriseByCondition(1:479),D_KL(2:480));


    % scatter(surpriseByCondition(1:479),D_KL)
    D_KL=(D_KL-min(D_KL))./(max(D_KL)-min(D_KL));
    I=(I-min(I))./(max(I)-min(I));

    for m=1:2000
        % terms:
        % 1 = intercept
        [b,bint,r,rint,stats] = regress(y2(:,m),[ones(size(surprise)) surprise-mean(surprise), isCPCondition-mean(isCPCondition),  surpriseByCondition-mean(surpriseByCondition)]);
        %[b,bint,r,rint,stats] = regress(y2(:,m),[ones(size(surprise)) I-mean(I), isCPCondition-mean(isCPCondition),  D_KL-mean(D_KL)]);       
        coefficient(m,simSub)=b(2);
        intercept(m,simSub)=b(1);
        condEffect(m,simSub)=b(3);
        surpByCond(m,simSub)=b(4);
    end
    aveOddballResp=mean(y2(1:480,:),2);
    aveChangeResp=mean(y2(481:end,:),2);
    scatter(surprise(481:end),aveChangeResp)
    
end

hold on
[h1,hp]=boundedline(1:2000,mean(coefficient,2),std(coefficient,0,2));
h1.Color=[0.8 0.8 0.8];
h1.LineWidth=1.5;
hp.FaceColor=[0.9 0.9 0.9];
[h2,hp2]=boundedline(1:2000,mean(surpByCond,2),std(surpByCond,0,2));
hp2.FaceColor=[0.85 0.85 0.85];
h2.Color=[0.6 0.6 0.6];
h2.LineWidth=1.5;
plot([0 2000],[0 0],'--k')
ylim([-0.003 0.003])
ylabel('coefficinets')
xlabel('time')
str = {'UPDATE','SURPRISE'};
t=text([1800 1800],[-0.001 0.001],str);
t(1).Color=[0.6 0.6 0.6];
t(2).Color=[0.8 0.8 0.8];
hold off
