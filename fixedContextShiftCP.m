
%%
% This code simulates 100 runs of the changepoint (CP) condition for each fixed context
% shift model to produce the effective learning rate corresponding to each
% particular context shift:

%% ------------------------------------------------------------------------
% Parameters:

numTrials                = 480;
Mu                       = nan(1,numTrials); % outcome means
Haz                      = 0.1; % Hazard Rate
%noiseStd                 = [25 * ones(120,1); 10*ones(120,1); 25 * ones(120,1); 10*ones(120,1)];
noiseStd                 = 25.*ones(480,1);
numReps                  = 100;
dropTrials               = 10;


% Neural Network:
contextMeans             = -pi:0.1:pi;% "context layer" with VM tuned neurons
contextConc              = 32;%PDF concentration
contextStartValue        = 0;
outputMeans              = -50:10:350;% "Output Layer" with Gaussian units
outputStd                = noiseStd;
fixedLR                  = 0.1;
FG                       = 0.1; % multiplier by which unused context weights are scaled for forgetting
actThresh                = 0.01; % threshold for forgetting weights
useLinearReadout         = true; % Readout of context layer. Default: True
shiftIncs                = linspace(0,1,50);
nModels                  = length(shiftIncs); % number of models
contextShift             = zeros(nModels,numTrials);
actThreshold             = 0.0001;
% Initialize structures to store results:
bLearningRate            = zeros (nModels,numReps); % Empirical (behavioral) learning rate

%% ------------------------------------------------------------------------
% Simulate:

for h=1:numReps
    
    % Initialize structures for one simulation:
    TAC=zeros(numTrials,1);
    outcome=nan(numTrials,1);
    
    % Loop through trials for generating outcomes:
    for n = 1:numTrials
        if n==1 || rand <Haz
            mu=rand.*250+25;
            TAC(n)=0;
        else
            TAC(n)=TAC(n-1)+1;
        end
        outcome(n)=round(normrnd(mu, noiseStd(n)));
    end
    
    outcome(outcome>300)=300;
    outcome(outcome<0)=0;
    
    
   
    for j = 1:50
        
        % Initialize random weights matrix:
        weightMatrix =(rand(length(outputMeans),...
            length(contextMeans))-.5)./1; % currently initializing to positive weights -- should try balanced weights too.
        
        % Initialize estimates of the network:
        estimate=nan(size(outcome));
        contextValue= contextStartValue;
        
        % Initialize context layer to fire:
        contextAct=circ_vmpdf(contextValue, contextMeans, contextConc);
        contextAct=contextAct./sum(contextAct);
                
     
        
        %Loop through trials to get network responses:
        for i = 1:length(outcome)
            
            % Get summed inputs for output layer:
            outputAct=contextAct*weightMatrix'; % Produce a response
            outputAct(outputAct<actThreshold)=0; % Currently the only non-linearity is to get rid of negative firing rates
            outputAct=outputAct./(sum(outputAct));
            
            % Linear readout of output neurons:
            if   useLinearReadout
                estimate(i)=outputMeans* (outputAct)'; % Normalize activity...
            else %   Readout maximum
                estimate(i)=outputMeans(find(outputAct==max(outputAct),1));
                
            end
        
            contextShift(j,i)  =shiftIncs(j);
            
            % Shift the context value :
            contextValue=contextValue+contextShift(j,i);
            
            % Set context layer activity based on new context value.
            contextAct=circ_vmpdf(contextValue, contextMeans, contextConc);
            contextAct=contextAct./sum(contextAct);
       
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
        
        % Compute (behavioral) learning rate :
        xes=[ones(size(PE)),PE];
        C = regress(UP(dropTrials:end),xes(dropTrials:end,:));
        bLearningRate(j,h)=C(2);
        
    end

end

simulatedLearningRateCP = mean(bLearningRate,2);
save('simulatedLearningRateCP','simulatedLearningRateCP')