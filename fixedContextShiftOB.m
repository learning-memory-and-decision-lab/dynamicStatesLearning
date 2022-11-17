

%% ------------------------------------------------------------------------

%% ------------------------------------------------------------------------
% Parameters:

%  Simulation:
numTrials                = 480;
Mu                       = nan(1,numTrials); % outcome means
Haz                      = 0.1; % Hazard Rate
driftRate                = 10;
noiseStd                 = 25;
numReps                  = 100;
dropTrials               = 10;
showPlots                = false;
k                        = 10; % Show PE for k trials after odd ball
% Neural Network:
contextMeans             = -pi:0.01:pi;% "context layer" with VM tuned neurons
contextConc              = 16;%PDF concentration
contextStartValue        = 0;
outputMeans              = -50:10:350;% "Output Layer" with Gaussian units
outputStd                = 25;
fixedLR                  = 0.1; % Neural Network fixed learning rate
FG                       = 0.1; % multiplier by which unused context weights are scaled for forgetting
actThresh                = 0.01; % threshold for forgetting weights
useLinearReadout         = true; % Readout of context layer. Default: True
weightScale              = 10e-1;
shiftIncs                = linspace(0,2,50); %  Fixed context shifts
nModels                  = length (shiftIncs);
contextShift             = zeros (nModels,numTrials);

% Initialize results structures:
bLearningRate            = zeros (nModels,numReps); % Empirical (behavioral) learning rate

%% ------------------------------------------------------------------------
% Simulate:

for h=1:numReps
    
    % Initialize structures for one simulation:
    TAC              = ones(numTrials,1); % trial after odd ball
    outcome          = nan(numTrials,1);
    
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
            
        elseif n < numTrials
            TAC(n+1)=TAC(n)+1;
        end
        
    end
    
    outcome(outcome>300)=300;
    outcome(outcome<0)=0;
    
    
    % Loop through different context shifts:
    for j =1: nModels
        % Initialize random weights matrix:
        weightMatrix=(rand(length(outputMeans),...
            length(contextMeans))-.5)./1000; % currently initializing to positive weights -- should try balanced weights too.

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
            
            % Linear readout of output neurons:
            if useLinearReadout
                estimate(i)=outputMeans* (outputAct)'; % Normalize activity...
            else %   readout maximum
                estimate(i)=outputMeans(find(outputAct==max(outputAct), 1));
            end
          
            
            % Now that we have generated a response, shift the context:
      
            % Set the context shift size:
      
            contextShift(j,i)  =shiftIncs(j);
            
            % Shift the context value :
            contextValue=contextValue+contextShift(j,i);
            
            % Set context layer activity based on new context value.
            contextAct=circ_vmpdf(contextValue, contextMeans, contextConc);
            contextAct=contextAct./sum(contextAct);
            
            % Provide supervised signal for learning:
            targetActivation=normpdf(outcome(i), outputMeans, outputStd); % we're giving model some info about noise
            targetActivation=targetActivation./(sum(targetActivation));
            normAdjusts=(targetActivation)'*contextAct;
            
            % Update weights:
            weightMatrix=weightMatrix.*(1-fixedLR) +fixedLR.* normAdjusts;
            
            % Forget inactive weights:
            weightMatrix(:, contextAct<actThresh)=...
                weightMatrix(:,contextAct<actThresh).*FG;
            
            
         
        end
        
        %  Result of the simulation:
        newBlock=false(size(outcome));
        newBlock(1)=true;
        [~, UP, PE]=computeLR(outcome', estimate, newBlock');
        
        % Compute Learning Rate for one simulation: "UP = Updates" as a funtion of
        % "PE = Prediction Errors"
        xes=[ones(size(PE)),PE];
        C = regress(UP(dropTrials:end),xes(dropTrials:end,:));
        bLearningRate(j,h)=C(2);
                
    end
      

end
simulatedLearningRateOB=mean(bLearningRate,2);
save('simulatedLearningRateOB.mat')