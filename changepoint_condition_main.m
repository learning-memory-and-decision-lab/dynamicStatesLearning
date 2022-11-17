
%clear; close all;


%% ------------------------------------------------------------------------
% Run learnigRateSimulation
if exist('simulatedLearningRateCP.mat')

    load('simulatedLearningRateCP.mat');
else
    fixedContextShiftCP
end

% Run humanData to get data for Figure 4:
humanData



%% ------------------------------------------------------------------------
% Parameters:

numTrials                = 480;
Mu                       = nan(1,numTrials); % outcome means
Haz                      = 0.1; % Hazard Rate

% Alernating noise levels across blocks ( same as the subject dataset)
%noiseStd                = [25 * ones(120,1); 10*ones(120,1); 25 * ones(120,1); 10*ones(120,1)];
noiseStd                 = 25.*ones(480,1); % Fixed noise
numReps                  = 32;
dropTrials               = 10;
showPlots                = true;
k                        = 11 ; % trials after a change point


% Neural Network:
contextMeans             = -pi:0.1:pi;% "context layer" with VM tuned neurons
contextConc              = 32;%PDF concentration
contextStartValue        = 0;
outputMeans              = -50:10:350;% "Output Layer" with Gaussian units
outputStd                =  noiseStd;


fixedLR                  = 0.1; % Neural network learning rate
FG                       = 0.1; % multiplier by which unused context weights are scaled for forgetting
actThresh                = 0.01; % threshold for forgetting weights
useLinearReadout         = true; % Readout of context layer. Default: True
shiftIncs                = linspace(0,1,50);
nModels                  = length(shiftIncs)+3; % number of models
contextShift             = zeros(nModels,numTrials);

% Initialize structures to store results:
bLearningRate            = zeros(nModels,numReps); % Effective learning rate
absNonCP_PE              = zeros(nModels,numReps); % Non-change point trials errors
nonstableError           = zeros(nModels,numReps); % Errors during nonstable period
stableError              = zeros(nModels,numReps); % Errors during stable period
delta_PE                 = zeros(nModels,numReps);

%% ------------------------------------------------------------------------
% Simulate

for jj=6 % try different hazard rates . Currently set to 0.7
    
    hazRate=0.1*jj;

    % numReps represents different subjects
    for h=1:numReps

        % Initialize structures for one simulation:
        TAC=zeros(numTrials,1);
        outcome=nan(numTrials,1);
        groundTruthContext=zeros(1,numTrials);

        % Loop through trials for generating outcomes:
        for n = 1:numTrials
            if n==1 || rand <Haz
                mu=rand.*250+25;
                TAC(n)=0;
                groundTruthContext(n)= nan;
            else
                TAC(n)=TAC(n-1)+1;
            end
            outcome(n)=round(normrnd(mu, noiseStd(n)));
        end

        outcome(outcome>300)=300;
        outcome(outcome<0)=0;
        feedbackChange=groundTruthContext;
        feedbackChange(isnan(groundTruthContext))=1;

        % Set ground truth context shift at change points euqal. to LR ==1 :
        groundTruthContext(isnan(groundTruthContext))=LR2shift(shiftIncs',simulatedLearningRateCP,1);


        % Use normative model to compute CPP, RU & (model)LR:
        B = nan(size(outcome))'; pCha = B; R = pCha;

        % Loop through noise levels: 25-10-25-10
        for i=1:4
            [B1, totSig1, R1, pCha1, ~] = frugFun5(outcome((i-1)*120+1:i*120)',hazRate, noiseStd(i*120), 0, 1, false...
                , 150, .1);
            R(1,(i-1)*120+1:i*120)= R1(1:end-1);
            pCha(1,(i-1)*120+1:i*120)= pCha1;
            B(1,(i-1)*120+1:i*120)=B1(1:end-1);
            totSig(1,(i-1)*120+1:i*120)=totSig1(1:end);

        end

        RU=1./(R+1);
        modelLR=pCha;

        % Loop through different context shifts:
        for j =  1:nModels

            % Initialize random weights matrix:
            weightMatrix =(rand(length(outputMeans),...
                length(contextMeans))-.5)./1000; % currently initializing to positive weights -- should try balanced weights too.
            % Initialize random weights matrix:
            initialWeightMatrix =(rand(length(outputMeans),...
                length(contextMeans))-.5)./1000; % currently initializing to positive weights -- should try balanced weights too.

            % weightMatrix=normrnd(0,0.5,length(outputMeans),length(contextMeans))./1000;

            % Initialize estimates of the network:
            estimate=nan(size(outcome));
            deltaEstimate=estimate;
            contextValue= contextStartValue;

            % Initialize context layer to fire:
            contextAct=circ_vmpdf(contextValue, contextMeans, contextConc);
            contextAct=contextAct./sum(contextAct);
            % Delta rule model comparison- only for fixed context shifts
            deltaOutputAct=contextAct*weightMatrix';
            if j<size(simulatedLearningRateCP,1)+1
                deltaLR=simulatedLearningRateCP(j);
            end

            %Loop through trials to get network responses:
            for i = 1:length(outcome)

                % Get summed inputs for output layer:
                outputAct=contextAct*weightMatrix'; % Produce a response
                outputAct(outputAct<0.0001)=0; % Currently the only non-linearity is to get rid of negative firing rates
                outputAct=outputAct./(sum(outputAct));

                % interpolate outcome to average of nearset output nodes:
                interOutProb=IDW(outcome(i),outputAct);

                % Compute network estimate of changepoint probability
                networkCPP(i,1)=((1/41) * hazRate) ./((interOutProb)*(1-hazRate) +(1/41) *hazRate);


                % Linear readout of output neurons:
                if   useLinearReadout
                    estimate(i)=outputMeans* (outputAct)'; % Normalize activity...
                    deltaEstimate(i)=outputMeans * (deltaOutputAct)';
                else %   Readout maximum
                    estimate(i)=outputMeans(find(outputAct==max(outputAct),1));

                end


                % Set Context shift according to model type:
                if j==length(shiftIncs)+1 % Ground Truth model

                    contextShift(j,i)= groundTruthContext(i);

                elseif j==length(shiftIncs)+2 % Dynamic context shift model

                    contextShift(j,i)=LR2shift(shiftIncs',simulatedLearningRateCP,modelLR(i));

                elseif j==length(shiftIncs)+3 % Dynamic context shift model
                    % convert network-estimated CPP to a context shift

                    contextShift(j,i)=LR2shift(shiftIncs',simulatedLearningRateCP,networkCPP(i));
                    changeUpdate(h,i) = contextShift(j,i);

                else %Fixed context shift model
                    contextShift(j,i)  =shiftIncs(j);
                end

                % Shift the context value :
                contextValue=contextValue+contextShift(j,i);


                % Set context layer activity based on new context value.
                contextAct=circ_vmpdf(contextValue, contextMeans, contextConc);
                contextAct=contextAct./sum(contextAct);

                if j==53
                    % Store dynamic context shift model context for dissimilarity matrix:
                    x=size(contextMeans,2)*floor(contextValue/(2.*pi));
                    totalContext(i,x+1:x+size(contextMeans,2))= contextAct;
                end

                % Provide supervised signal for learning:
                targetActivation=normpdf(outcome(i), outputMeans,outputStd(i)); % we're giving model some info about noise
                targetActivation=targetActivation./(sum(targetActivation));
                normAdjusts=(targetActivation)'*contextAct;

                % Update weights:
                weightMatrix=weightMatrix.*(1-fixedLR) +fixedLR.* normAdjusts;

                % Forget inactive weights:
                weightMatrix(:, contextAct<actThresh)=...
                    weightMatrix(:,contextAct<actThresh).*FG;

            end

            % Compute Prediction Error and Update:
            newBlock=false(size(outcome));
            newBlock(1)=true;
            [~, UP, PE]=computeLR(outcome', estimate', newBlock');

            % Compute (effective) learning rate :
            xes=[ones(size(PE)),PE];
            C = regress(UP(dropTrials:end),xes(dropTrials:end,:));
            bLearningRate(j,h)=C(2);


            % Learning rate according to trial after CP:
            for trialBin=1:k
                prob=TAC==trialBin-1;
                C2 = regress(UP(prob),xes(prob,:));
                trialLR(j,trialBin)=C2(2);
            end

            
            a=[0  0.25 0.5 0.75 2];
            for cpBin=1:length(a)-1
                prob=and(a(cpBin+1)>contextShift(52,:),a(cpBin)<=contextShift(52,:))';
                C2 = regress(UP(prob),xes(prob,:));
                changePointLR(cpBin)=C2(2);
            end


            % Does neural network model adjust beliefs like Bayesian model?
            [LR_norm, UP_norm, PE_norm]=computeLR(outcome', B, newBlock');

            % Does neural network model adjust learning according to RU & CPP
            xMat=[ones(size(PE(dropTrials:end-1))), PE(dropTrials:end-1),...
                PE(dropTrials:end-1).* (pCha(dropTrials:end-1)-...
                mean(pCha(dropTrials:end-1)))', PE(dropTrials:end-1).*...
                (RU(dropTrials:end-1)'- mean(RU(dropTrials:end-1),2))];
            b_nn(:,j)=regress(UP(dropTrials:end-1), xMat); % neural net coefficients -- is CPP coefficient positive and large (eg. close to 1)

            % Does normative model adjust learning according to RU & CPP
            xMat_norm=[ones(size(PE_norm(dropTrials:end-1))),...
                PE_norm(dropTrials:end-1), PE_norm(dropTrials:end-1).*...
                (pCha(dropTrials:end-1)-mean(pCha(dropTrials:end-1)))',...
                PE_norm(dropTrials:end-1).*(RU(dropTrials:end-1)'- ...
                mean(RU(dropTrials:end-1),2))];
            b_norm=regress(UP_norm(dropTrials:end-1), xMat_norm);

            % Store Prediction Error of non-change point trials:
            absNonCP_PE(j,h)=nanmean(abs(outcome(TAC>0)-estimate(TAC>0)));
            nonstableError(j,h)=nanmean(abs(outcome(TAC==1)-estimate(TAC==1)));
            stableError(j,h)=nanmean(abs(outcome(TAC>5)-estimate(TAC>5)));
            delta_PE(j,h)=nanmean(abs(outcome(TAC>0)-deltaEstimate(TAC>0)));

        end
        fixedTrialLR(h,:)= (trialLR(25,:));
        groundTruthTrialLR(h,:)=trialLR(51,:);
        DynamicTrialLR(h,:)=trialLR(52,:);
        networkTrialLR(h,:)=trialLR(53,:);

        b_nnG(:,h)=b_nn(:,51);
        b_nnD(:,h)=b_nn(:,52);
        b_nnN(:,h)=b_nn(:,53);

        % Store feedback for pupil response simulations:
        feedBackChangeCond(h,:)=feedbackChange;
        changeProbCond(h,:)=networkCPP;
        changeMean(h,:)=B;
        changeSTD(h,:)=totSig;
        contextShiftLRCP(h,:) = changePointLR;
        



    end
    error(jj,:)=absNonCP_PE(52,:);
    error_net(jj,:)=absNonCP_PE(53,:);
    error_fixed(jj,:)=absNonCP_PE(11,:);

end

%save('contextShiftLRCP','contextShiftLRCP')

% save variables for pupil response simulations:
save('pCha.mat','changeProbCond')
save('cSTD.mat','changeSTD')
save('feedbackChange.mat','feedBackChangeCond')
save('cMean.mat','changeMean')
save('changeUpdate.mat','changeUpdate')
save('contextShiftLRCP.mat','contextShiftLRCP')



%% ------------------------------------------------------------------------
% Plots:

if showPlots

    figure;
    hold on
    boundedline(shiftIncs(1:25),absNonCP_PE(1:25),std(absNonCP_PE(1:25,:),0,2),'--');
    errorbar(mean(contextShift(51,:)),mean(absNonCP_PE(end-2,:),2),std(absNonCP_PE(51,:),0,2),'.','MarkerSize',25)
    errorbar(mean(contextShift(52,:)),mean(absNonCP_PE(end-1,:),2),std(absNonCP_PE(52,:),0,2),'.','MarkerSize',25)
    errorbar(mean(contextShift(53,:)),mean(absNonCP_PE(end,:),2),std(absNonCP_PE(53,:),0,2),'.','MarkerSize',25)
    legend('Standard Deviation','Fixed context shift','Ground Truth model','Dynamic context shift model','Network-based Model')
    ylabel('Error')
    xlabel('Average Context Shift')
    title('Error on Average')

    figure;
    hold on
    boundedline(shiftIncs(1:50),mean(nonstableError(1:50,:),2),std(nonstableError(1:50,:),0,2),'--');
    errorbar(mean(contextShift(51,TAC==0)),mean(nonstableError(end-2,:),2),std(nonstableError(51,:),0,2),'.','MarkerSize',25)
    errorbar(mean(contextShift(52,TAC==0)),mean(nonstableError(end-1,:),2),std(nonstableError(52,:),0,2),'.','MarkerSize',25)
    errorbar(mean(contextShift(53,TAC==0)),mean(nonstableError(end,:),2),std(nonstableError(53,:),0,2),'.','MarkerSize',25)
    legend('Standard Deviation','Fixed context shift','Ground Truth model','Dynamic context shift model','Network-based Model')
    ylabel('Error')
    xlabel('Average Context Shift')
    title('Error on ChangePoints')
    ylim([10 80])


    figure;
    hold on
    boundedline(shiftIncs(1:25),stableError(1:25),std(stableError(1:25,:),0,2),'--');
    errorbar(mean(contextShift(51,TAC>4)),mean(stableError(end-2,:),2),std(stableError(51,:),0,2),'.','MarkerSize',25)
    errorbar(mean(contextShift(52,TAC>4)),mean(stableError(end-1,:),2),std(stableError(52,:),0,2),'.','MarkerSize',25)
    errorbar(mean(contextShift(53,TAC>4)),mean(stableError(end,:),2),std(stableError(53,:),0,2),'.','MarkerSize',25)
    legend('Standard Deviation','Fixed context shift','Ground Truth model','Dynamic context shift model','Network-based Model')
    ylabel('Error')
    xlabel('Average Context Shift')
    title('Error in Stable Period')
    xlim([-0.05 1])

    figure;
    hold on
    boundedline(shiftIncs(1:25),mean(bLearningRate(1:25,:),2),std(bLearningRate(1:25,:),0,2),'--');
    errorbar(mean(contextShift(51,TAC>0)),mean(bLearningRate(end-2,:),2),std(bLearningRate(51,:),0,2),'.','MarkerSize',25)
    errorbar(mean(contextShift(52,TAC>0)),mean(bLearningRate(end-1,:),2),std(bLearningRate(52,:),0,2),'.','MarkerSize',25)
    errorbar(mean(contextShift(53,TAC>0)),mean(bLearningRate(end,:),2),std(bLearningRate(53,:),0,2),'.','MarkerSize',25)
    %legend('Standard Deviation','Fixed context shift','Ground Truth model','Dynamic context shift model','Netwrok-based context shifts')
    ylabel('Learning Rate')
    xlabel('Average Context Shift')
    xlim([-0.05 1])

    figure;
    hold on
    plot(1:11,mean(fixedTrialLR),'LineWidth',3);
    plot(1:11,mean(groundTruthTrialLR),'LineWidth',3);
    plot(1:11,mean(DynamicTrialLR),'LineWidth',3);
    a=plot(1:11,mean(networkTrialLR),'LineWidth',3);
    boundedline(1:11,mean(humanSubCE),std(humanSubCE)/sqrt(numReps));
    legend('Fixed context shift','Ground Truth Model','Dynamic context shift model','network-based model','Human Data')
    uistack(a,'top') %you can also do uistack(b,'bottom')
    ylabel('Learning Rate')
    xlabel('Trials after an changepoint')
    xlim([1 11])

    figure;
    hold on
    plot([-300 300],[0 0],'LineStyle','--','Color',[0 0 0.2],'LineWidth',2)
    plot([-300 300],[-300 300],'LineStyle','--','Color',[0 0 0.2],'LineWidth',2)
    s=scatter(PE(10:end),UP(10:end),25,pCha(10:end),'filled');
    s.MarkerFaceAlpha =1;
    c=colorbar;
    colormap((copper))
    c.Label.String = 'Change Point Probability';
    xlabel('Prediction Error','FontSize',14,'FontWeight','bold')
    ylabel('Update','FontSize',14,'FontWeight','bold')

    figure;
    hold on
    plot([-300 300],[0 0],'LineStyle','--','Color',[0 0 0.2],'LineWidth',2)
    plot([-300 300],[-300 300],'LineStyle','--','Color',[0 0 0.2],'LineWidth',2)
    s=scatter(PE(10:end),UP(10:end),25,RU(10:end),'filled');
    s.MarkerFaceAlpha =1;
    c=colorbar;
    colormap((copper))
    c.Label.String = 'Relative Uncertainty';
    xlabel('Prediction Error','FontSize',14,'FontWeight','bold')
    ylabel('Update','FontSize',14,'FontWeight','bold')


    figure;
    hold on
    for i=1:32
        d=scatter(1,bSubs(i,2),'filled','MarkerFaceColor',[0.8 0.8 0.8],'jitter','on','jitterAmount',0.2);
        scatter(2,bSubs(i,3),'filled','MarkerFaceColor',[0.8 0.8 0.8],'jitter','on','jitterAmount',0.2);
        scatter(3,bSubs(i,4),'filled','MarkerFaceColor',[0.8 0.8 0.8],'jitter','on','jitterAmount',0.2);

    end

    plot(1,b_nn(2,25),'o','MarkerFaceColor','g','MarkerSize',8)
    plot(2,b_nn(3,25),'o','MarkerFaceColor','g','MarkerSize',8)
    plot(3,b_nn(4,25),'o','MarkerFaceColor','g','MarkerSize',8)

    a=errorbar(1,mean(b_nnG(2,:)),std(b_nnG(2,:))./sqrt(numReps),'.','Color','r','MarkerSize',25)
    b=errorbar(1,mean(b_nnD(2,:)),std(b_nnD(2,:))./sqrt(numReps),'.','Color','y','MarkerSize',25)
    c=errorbar(1,mean(b_nnN(2,:)),std(b_nnN(2,:))./sqrt(numReps),'.','Color','m','MarkerSize',25)


    errorbar(2,mean(b_nnG(3,:)),std(b_nnG(3,:))./sqrt(numReps),'.','Color','r','MarkerSize',25)
    errorbar(2,mean(b_nnD(3,:)),std(b_nnD(3,:))./sqrt(numReps),'.','Color','y','MarkerSize',25)
    errorbar(2,mean(b_nnN(3,:)),std(b_nnN(3,:))./sqrt(numReps),'.','Color','m','MarkerSize',25)


    a=errorbar(3,mean(b_nnG(4,:)),std(b_nnG(4,:))./sqrt(numReps),'.','Color','r','MarkerSize',25)
    b=errorbar(3,mean(b_nnD(4,:)),std(b_nnD(4,:))./sqrt(numReps),'.','Color','y','MarkerSize',25);
    c=errorbar(3,mean(b_nnN(4,:)),std(b_nnN(4,:))./sqrt(numReps),'.','Color','m','MarkerSize',25);

    legend([d a b c],{'Human Data','GroundTruth','Bayesian Context Shift','Network Context Shift'})
    plot([0 5],[0 0],'--k')

    xlim([0 5])
    ylabel('coefficient')
    ylabel('term')
    hold off
end
