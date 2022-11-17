% Code for extracting human subjects responses(McGuire&Nassar2014) coefficients for PE ,
% PE.*(mean(pCha)-pCha) & RU.*(mean(RU)-RU), used in figure 4
%
% Output : bSubs (coefficients)

load('McGuireNassar2014data.mat')

noise=25; % outcome std
dropTrials=10; % discard first 10 trials
nSubs=max(allDataStruct.subjNum); %Number of subjects

% Initialize coefficients:
bSubs = nan(nSubs,4);
for i =1:nSubs
    
    sel=(allDataStruct.subjNum==i);
    Haz=allDataStruct.currentHazard(sel);
    prediction = allDataStruct.currentPrediction(sel);
    outcome = allDataStruct.currentOutcome(sel);
    % Remove JS failures:
    bagLoc=allDataStruct.currentOutcome(sel);
    tPerBlock=cat(1,allDataStruct.blockCompletedTrials(sel));
    newBlock=tPerBlock<=1;
    tol=30; % this is the free parameter... higher tolerences identify fewer trials.
    
    aMat=[nan nan; prediction(1:end-1) bagLoc(1:end-1)]; % this is a list of prev pred/outcomes
    jsFail=~newBlock & ((prediction < (min(aMat, [], 2)-tol) |  prediction > (max(aMat, [], 2)+tol))); % we're way out of that range, call it a joystick fail.
    
    
    % Use normative model to compute CPP, RU & (model)LR:
    
    %[B, totSig, R, pCha, ~] = frugFun5(outcome, 0.125, 25, 0, 1, false...
    %    , 150, .1);
     
    % Use normative model to compute CPP, RU & (model)LR:
    B = nan(size(outcome))'; pCha = B; R = pCha;
    % Loop through noise levels: 25-10-25-10
    currentSubjectNoise=allDataStruct.blockStds(allDataStruct.subjNum==i);
    blockNum=allDataStruct.blkNum(allDataStruct.subjNum==i);
    numbTrial(1)=0;
    for j=1:4
        noiseStd=mean(currentSubjectNoise(blockNum==j));
        blockOutcome=outcome(blockNum==j);
        numbTrial(j+1)=numbTrial(j)+size(blockOutcome,1);
        [B1, totSig, R1, pCha1, ~] = frugFun5(blockOutcome',0.1,noiseStd , 0, 1, false...
            , 150, .1);
        R(1,numbTrial(j)+1:numbTrial(j+1))= R1(1:end-1);
        pCha(1,numbTrial(j)+1:numbTrial(j+1))= pCha1;
        B(1,numbTrial(j)+1:numbTrial(j+1))=B1(1:end-1);
    end
    [~, UP, PE]=computeLR(outcome', prediction', newBlock');
    
    
    RU=1./(R(1:end)+1);
    PE=PE(jsFail==0);
    RU=RU(jsFail==0);
    pCha=pCha(jsFail==0);
    UP=UP(jsFail==0);
    
    xMat=[ones(size(PE(dropTrials:end-1))),...
        PE(dropTrials:end-1), PE(dropTrials:end-1).*...
        (pCha(dropTrials:end-1)-mean(pCha(dropTrials:end-1)))',...
        PE(dropTrials:end-1).*(RU(dropTrials:end-1)'- ...
        mean(RU(dropTrials:end-1),2))];
    belief=B(1:end)';
    belief=belief(jsFail==0);
    bSubs(i,:)=regress(UP(dropTrials:end-1), xMat);
    
    UP=nan(length(outcome),1);   %nans
    UP(1:end-1)=prediction(2:end)-prediction(1:end-1);
    PError=(outcome-prediction);
    xes=[ones(size(PError)),PError];
    cp=allDataStruct.isChangeTrial(sel);
    clear TAC
    TAC(1)=0;
    
    for j=2:length(cp)
        if cp(j)==1
            TAC(j)=0;
        else
            TAC(j)=TAC(j-1)+1;
        end
    end
    TAC=TAC';
    
    for t=1:11
        prob=TAC==t-1;
        C2 = regress(UP(prob),xes(prob,:));
        humanSubCE(i,t)=C2(2);
    end
end
save('bSubs.mat');
save('humanSubCE');

