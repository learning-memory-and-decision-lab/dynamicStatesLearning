
clear; close all;

%% ------------------------------------------------------------------------
if exist('simulatedLearningRateOB.mat')
    load('simulatedLearningRateOB.mat');
else

    fixedContextShiftOB
end

if exist('contextShiftLRCP.mat')
    load('contextShiftLRCP.mat')
else
    changepoint_condition_main
end


%% ------------------------------------------------------------------------
% Parameters:

%  Simulation:
numTrials                = 480;
Mu                       = nan(1,numTrials); % outcome means
Haz                      = 0.1; % Hazard Rate
driftRate                = 10;
noiseStd                 = 25;
numReps                  = 32;
dropTrials               = 10;
showPlots                = true;
k                        = 11; % Show PE for k trials after odd ball

% Neural Network:
contextMeans             = -pi:0.1:pi;% "context layer" with VM tuned neurons
contextConc              = 32;%PDF concentration
contextStartValue        = 0;
outputMeans              = -50:10:350;% "Output Layer" with Gaussian units
outputStd                = 25;
fixedLR                  = 0.1; % Neural Network fixed learning rate
FG                       = 0.1; % multiplier by which unused context weights are scaled for forgetting
actThresh                = 0.01; % threshold for forgetting weights
useLinearReadout         = true; % Readout of context layer. Default: True
weightScale              = 10e-1;
shiftIncs                = linspace(0,1,50); %  Fixed context shifts
nModels                  = length (shiftIncs)+4;
contextShift             = zeros (nModels,numTrials);
OddLRs                   = simulatedLearningRateOB;
% Initialize results structures:
bLearningRate            = zeros (nModels,numReps); % Empirical (behavioral) learning rate
absNonCP_PE              = zeros(nModels,numReps);
allEstimate              = nan(nModels,numTrials);
trialPE                  = nan(nModels,k);

%% ------------------------------------------------------------------------
% Simulate:

for jj=0.6

    hazRate=0.1*jj;

    for h=1:numReps

        % Initialize structures for one simulation:
        TAC              = zeros(numTrials,1); % trial after odd ball
        outcome          = nan(numTrials,1);
        errBased_RU      = [0.9 0.9];
        groundTruthContext            = zeros(1,numTrials); % Ground truth context shifts

        % Loop through trials for generating outcomes:
        for n = 1:numTrials

            % Choose mean:
            if n ==1
                Mu(n)=rand.*100+ 100; % start helicopter in the middlish region
            else
                Mu(n)=Mu(n-1)+normrnd(0, driftRate);
            end

            % make boundary reflective
            if Mu(n)>300
                Mu(n) = 300 -  (Mu(n)-300);
            elseif Mu(n)<0
                Mu(n)=abs(Mu(n));
            end

            %Create outcome:
            outcome(n)=round(normrnd(Mu(n), noiseStd));

            %Add Oddballs:
            if rand <Haz
                outcome(n)=floor(rand.*300);
                TAC(n)=0; % Odd ball
                groundTruthContext(n)= nan;

            elseif n>1
                TAC(n)=TAC(n-1)+1;
            end

        end

        outcome(outcome>300)=300;
        outcome(outcome<0)=0;

        % Feedback is stored for pupil response simulations
        feedbackOdd=groundTruthContext;
        feedbackOdd(isnan(groundTruthContext))=1;
        groundTruthContext(isnan(groundTruthContext))=LR2shift(shiftIncs',OddLRs,1);


        % Loop through different context shifts:
        for j = 1:nModels
            % Initialize random weights matrix:
            weightMatrix=(rand(length(outputMeans),...
                length(contextMeans))-.5)./1000; % currently initializing to positive weights -- should try balanced weights too.

            % Initialize estimates of the network:
            estimate=nan(size(outcome));

            % Initialize uncertainty and surprise vectors:
            oppLR = nan(size(outcome));
            modifiedLR = nan(size(outcome));

            contextValue= contextStartValue;

            % Initialize context layer to fire:
            contextAct=circ_vmpdf(contextValue, contextMeans, contextConc);
            contextAct=contextAct./sum(contextAct);

            %Loop through trials to get network responses:
            for i = 1:length(outcome)

                % Activity of context layer:
                contextAct=circ_vmpdf(contextValue, contextMeans, contextConc);
                contextAct=contextAct./sum(contextAct);

                % Get summed inputs for output layer:
                outputAct=contextAct*weightMatrix'; % Produce a response
                outputAct(outputAct<0)=0; % Currently the only non-linearity is to get rid of negative firing rates
                outputAct=outputAct./(sum(outputAct));


                interOutProb=IDW(outcome(i),outputAct);

                networkCPP(i,1)=((1/41) * hazRate) ./((interOutProb)*(1-hazRate) +(1/41) *hazRate);

                % Linear readout of output neurons:
                if useLinearReadout
                    estimate(i)=outputMeans* (outputAct)'; % Normalize activity...
                else %   readout maximum
                    estimate(i)=outputMeans(find(outputAct==max(outputAct), 1));
                end


                % Now that we have generated a response, shift the context:

                % How to compute our dynamic context shift :

                % Step 1 = compute prediction error
                PE=outcome(i)-estimate(i);

                % Step 2 = compute surprise and uncertainty according
                % to prediction error
                [errBased_pOdd,errBased_RU,errBased_LR,errBased_UP,totUnc]=...
                    getTrialVarsFromPEs_cannon([outputStd; outputStd],...
                    [PE; 0], hazRate, [1; 0], [false, false], ...
                    errBased_RU(2), 0, 1, 1, [driftRate; driftRate]...
                    , [true; true], 300);

                

                % Step 3 = compute learning rate according to
                % surprise and uncertainty:
                oppLR(i)=(errBased_pOdd(1)+errBased_RU(2))...
                    -errBased_pOdd(1)*errBased_RU(2);

                modifiedLR(i)=errBased_RU(2);
                -errBased_pOdd(1)*errBased_RU(2);

                % Set the context shift size:
                if     j==length(shiftIncs)+1 % Ground truth model

                    contextShift(j,i)= groundTruthContext(i);

                elseif j==length(shiftIncs)+2 %  dynamic changepoint context shift model

                    contextShift(j,i)=LR2shift(shiftIncs',OddLRs,oppLR(i));


                elseif j==length(shiftIncs)+3 % dyanmic bayesian context shift model

                    contextShift(j,i)=LR2shift(shiftIncs',OddLRs,networkCPP(i)');


                elseif j==length(shiftIncs)+4 % dynamic netowrk-based shift model

                    contextShift(j,i)=LR2shift(shiftIncs',OddLRs,networkCPP(i)');

                    % Store Bayesian mean and std for pupil response
                    % simulations:
                    B(i) = estimate(i) + errBased_UP(1);
                    totSig(i) = sqrt(totUnc);

                    % elseif j==length(shiftIncs)+4
                    %
                    %     contextShift(j,i)=LR2shift(shiftIncs',OddLRs,oppLR(i));

                else % Fixed context shift models
                    contextShift(j,i)  =shiftIncs(j);
                end

                % Shift the context value :
                contextValue=contextValue+contextShift(j,i);

                % Set context layer activity based on new context value.
                contextAct=circ_vmpdf(contextValue, contextMeans, contextConc);
                contextAct=contextAct./sum(contextAct);

                % Provide supervised signal for learning:
                targetActivation=normpdf(outcome(i), outputMeans, outputStd); % we're giving model some info about noise
                targetActivation=targetActivation./(sum(targetActivation));
                normAdjusts=(targetActivation)'*contextAct;
                %fixedLR=modifiedLR(i);
                % Update weights:
                weightMatrix=weightMatrix.*(1-fixedLR) +fixedLR.* normAdjusts;


                % Store context shifts before returning back the context:
                save_context(i,j)=contextShift(j,i);


                % Return back the context :
                if j== length(shiftIncs)+1 %Ground truth model

                    % Return back the context for next trial - add a constant
                    % drift
                    contextValue=contextValue-groundTruthContext(i)+0.05;
                   
                elseif j== length(shiftIncs)+3 % Dynamic bayesian context model
                    %Return back the context
                    contextValue = contextValue -contextShift(j,i);
                    save_context(i,j)=contextShift(j,i);

                    % Shift the context again
                    % Shift the context again
                    contextShift(j,i)= LR2shift(shiftIncs',OddLRs,modifiedLR(i));


                   elseif j== length(shiftIncs)+4 % Dynamic netwokr-based context model
                    %Return back the context
                    contextValue = contextValue -contextShift(j,i);
                    save_context(i,j)=contextShift(j,i);

                    % Shift the context again
                    % Shift the context again
                    contextShift(j,i)= LR2shift(shiftIncs',OddLRs,modifiedLR(i));
                    contextValue = contextValue + contextShift(j,i);
                    oddUpdate(h,i) = contextShift(j,i);

                    % Store oddball probability for the dynamic context shift
                    % model:
                    pOdd(i,1)=errBased_pOdd(1);
                    
                end
                % Forget inactive weights:
                weightMatrix(:, contextAct<actThresh)=...
                    weightMatrix(:,contextAct<actThresh).*FG;

            end

            %  Result of the simulation:
            newBlock=false(size(outcome));
            newBlock(1)=true;
            [~, UP, PE]=computeLR(outcome', estimate', newBlock');

            % Compute Learning Rate for one simulation: "UP = Updates" as a funtion of
            % "PE = Prediction Errors"
            xes=[ones(size(PE)),PE];
            C = regress(UP(dropTrials:end),xes(dropTrials:end,:));
            bLearningRate(j,h)=C(2);

            if j == length(shiftIncs)+4
                % store Prediction Error of k trials after odd ball:
                a=[0  0.25 0.5 0.75 1];
                for t=1:length(a)-1
                    %prob=and((a(t+1)>manLR),(a(t)<=manLR));
                    prob = TAC==t-1;
                    C2 = regress(UP(prob),xes(prob,:));
                    postOBtrialLR(t)=C2(2);
                end
            contextShiftLROB(h,:)= postOBtrialLR;

            end


            % Store Prediction Error of non-change point trials:
            absNonCP_PE(j,h)=nanmean(abs(outcome(TAC>0)-estimate(TAC>0)));
            allEstimate(j,:)=estimate';
        end

        % Store feedback and oddball probability for pupil response  simulations:
        feedBackOddCond(h,:)=feedbackOdd;
        oddMean(h,:)=B;
        oddSTD(h,:)=totSig;
        oddProbCond(h,:) =pOdd;

    end

end


save('contextShiftLROB','contextShiftLROB')


% save variables for pupil response simulations:
save('pOdd.mat','oddProbCond')
save('oddSTD.mat','oddSTD')
save('feedbackOdd.mat','feedBackOddCond')
save('oddMean.mat','oddMean')
save('oddUpdate.mat','oddUpdate')


%% ------------------------------------------------------------------------
% Plots:

if showPlots

    figure;
    hold on
    boundedline(shiftIncs(1:25),mean(absNonCP_PE(1:25,:),2),std(absNonCP_PE(1:25,:),0,2),'--');
    errorbar(mean(contextShift(51,:)),mean(absNonCP_PE(end-3,:)),std(absNonCP_PE(51,:),0,2),'.','MarkerSize',25)
    errorbar(mean(contextShift(52,:)),mean(absNonCP_PE(end-2,:)),std(absNonCP_PE(52,:),0,2),'.','MarkerSize',25)
    errorbar(mean(contextShift(53,:)),mean(absNonCP_PE(end-1,:)),std(absNonCP_PE(53,:),0,2),'.','MarkerSize',25)
    errorbar(mean(contextShift(54,:)),mean(absNonCP_PE(end,:)),std(absNonCP_PE(54,:),0,2),'.','MarkerSize',25)  
    legend('standard deviation','Fixed context shift','Ground Truth','dynamic changepoint model','dynamic bayesian model','Network-based context shift')
    ylabel('Error')
    xlabel('Average Context Shift')

    figure;
    hold on
    plot(outcome(1:60), 'or', 'markerFaceColor',[0.8 0.8 0.8], 'markerEdgeColor', 'k', 'lineWidth', 1, 'markerSize', 8)
    plot(allEstimate(53,1:60),'lineWidth',2)
    plot(allEstimate(52,1:60),'lineWidth',2)
    plot(allEstimate(54,1:60),'lineWidth',2)

    odd=find(TAC==0);
    hold on
    for i=1:7
        plot([(odd(i)) (odd(i))] ,[0 300],'--k')
    end
    hold off

    ylabel('Position')
    xlabel('Trial')
    title('Dynamic Context Shifts')
    l=legend('Outcome','dynamic bayesian learning rate ','dynamic changepoint','dynamic latent states');

end