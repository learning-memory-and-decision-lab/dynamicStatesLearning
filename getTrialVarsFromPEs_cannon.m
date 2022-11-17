function [errBased_pCha, errBased_RU, errBased_LR, errBased_UP,denominator]=getTrialVarsFromPEs_cannon...
    (noise, PE, modHaz, newBlock, allHeliVis,...
    initRU, heliVisVar, lw, ud, driftRate, isOdd, outcomeSpace)


% Inputs:
% noise     = standard deviation of guassian noise distribution
% PE        = trial by trial prediction errors
% modHaz    = hazard rate
% newBlock  = logical indicating where new blocks start
% allHeliVis= visible helicopter/cannon?
% initRU    = RU has to be initialized to something...
% heliVisVar= how reliable is visible helicopter/cannon info?
% lw        = likelihood weight for surprise sensitivity
% ud        = uncertainty depletion
% isOdd     = logical array defining whether trials are from oddball condition
% outcomeSpace = size of outcome space 


% OK, i'd like to set it up so that this model can actually make
% predictions if you want it to. The idea would be that you just send in a
% nan instead of the prediction error array.




% adapted by MRN from getTrialVarsFromPEs that was used in
% "berlinMastList_new" for aging paper.

% the main changes from the code must deal with:

% 1) the circular nature of the cannon task.
% 2) the the drift/oddball condition.  


% list of all changes made:

% 1) removed option ot compute CPP and RU in separate passes. All will be
% computed at once for this analysis. That is the more justifiable strategy
% and it worked better anyway for the aging data.

% 2) added a variable "outcomeSpace" that defines the range of possible
% outcomes. 

% 3) included a drift term that increases total uncertainty

% 4) ** WRITE UP UNCERTAINTY UPDATE EQUATIONS HERE~!!! **



% Original description of the code:



% This code can be used to get SUBJECTIVE estimates of change-point
% probability and relative uncertainty from 1) subject prediction errors
% and 2) a set of model parameters (hazard, heliVisVar, likelihood weight,
% Uncertainty Depletion). 

% NOTE: this code can compute CPP and RU in two different ways. If you set:
% separateCPP_RU to true, then the algorithm computes all values of CPP
% using some average value of RU. Then it goes through and computes RU
% values according to all of those values for CPP. This method guarentees
% that RU can't pick up any error magnitude related variance... but its not
% exactly accurate, because subjects are not actually maintaining a fixed
% value of RU across all trials (so the CPP computation is a bit wrong). 
% The second strategy comes when you set separateCPP_RU to false... in that
% regime you get sequential updating of CPP based on RU and RU based on the
% previous CPP. In most ways this should be more accurate because it
% accounts for differences in uncertainty, but it does suffer from the fact
% that it leaves some error related variance in the data (as CPP is no
% longer exactly monotonic in absolute errors). 



if nargin<11
    ud=1;
end

if nargin<10
    lw=1;
end



% OK, it turns out we're using this bit of code over and over. In order to
% prevent stuff from diverging at different parts of code I'm going to put
% it in a function.

% inputs:
% noise = array of trial standard deviations...
% PE= prediction errors from subject.
% modHaz: what hazard do you want to give to the model?
% separateCPP_RU: fit CPP and RU together (ie no assumptions about flat RU)
% relUnc is only necessary if you are separating CPP and RU estimation:
% newBlock: logical telling you where new blocks start
% allHeliVis: logical telling you whether heli was visible (zeros for no)
% initRU: starting RU value
% heliVisVar: how much variance on the visible helicopter predictive cue
%
% keyboard

errBased_RU   = nan(size(PE));
errBased_pCha = nan(size(PE));
errBased_LR   = nan(size(PE));
errBased_UP   = nan(size(PE));
H=modHaz;



for nn = 1:length(noise)
    
    
    
    % First compute relative uncertainty. 
    
    % IF this is the first trial in a block, set it to a fixed value:
    if newBlock(nn)
        errBased_RU(nn)=initRU;
    else
    % otherwise, compute it based on the previous trials prediction error.

        nVar=noise(nn-1).^2;
               
        cp   =errBased_pCha(nn-1);
        tPE  =PE(nn-1);
        inRU=errBased_RU(nn-1);
        
        
        
        
        
     % ok, this is complicated as shit. I'm pretty sure i've got it at this point, 
     % but the propagation of uncertainty, as it is written down here, is still not
     % totally intuitive. These notes are meant to provide some intuition: 
     
     % The key is that, if there is no change point or oddball, we want to
     % use the UPDATED value of uncertainty. That is to say: what is the
     % variance on the predictive distribution given the that we can use
     % the newest piece of information (and trust it completely). 
     
     % so as run length goes to infinity, this quantity goes to zero.
     % as run length goes to zero, this quantity goes to the noise variance
     % *NOTE* this is because we are assuming that we got a new piece of good info: 
     % so worst case scenario is we're at noise 
        
     % so through this lens it makes some sense that this should be the noise variance
     % divided by 1+run length . 
     
     % Why is this equal to inRU.*nVar ? Because inRU is equal to 1./(R+1).
     
     % OK, for the oddball condition, we want the oddball variance to be
     % equal to the total variance assuming that the new outcome provided
     % NO new information. This quantity should go to infinity as run length goes 
     % to zero... and toward zero as run length goes to infinity.      
     % It is 1/(run length)... 
        
     % given the intuition above, it is a bit easier to compute this update 
     % in terms of run length:
     
        runLength=(1-inRU)./inRU;
       
        if ~isOdd(nn)
        % if we are in the changepoint condition, update uncertainty as
        % we've done previously:   
        % from equation 6 in heliFMRI paper:
        numerator=(cp.*nVar)+((1-cp).*inRU.*nVar) + ...
            cp.*(1-cp).*(tPE.*(1-inRU)).^2;
        elseif isOdd(nn)  
        % difference in means = tPE.*inRU 
        % note, since oddballs are 0 learning, this is sort of opposite the
        % same term int the equation above (ie 1-inRU)
        
        % otherwise, update as if "cp" means oddball:
        numerator=(cp.*nVar./runLength)+((1-cp).*nVar./(runLength+1)) + ...
            cp.*(1-cp).*(tPE.*inRU).^2;
        
        % description for paper:
        % first term  = odd * (nVar*inRU)./(1-inRU )
        % second term = (1-odd)* inRU.*nVar
        % third term  = odd.*(1-odd).*(tPE.*inRU).^2;
        
        end
        
        
        
        
        
        
        
        % if we are in the drift condition, increment total uncertainty
        % according to the drift rate:
        if driftRate(nn)>0
        numerator=numerator+driftRate(nn).^2;
        end
        
        numerator=numerator./ud; % divide uncertainty about mean by constant
        denominator = numerator+nVar; % denominator is just numerator plus noise variacne
        errBased_RU(nn)=numerator./denominator; % RU is just the fraction

        % this is going to be skipped because we don't have catch trials:
        if allHeliVis(nn)==1
            % MRN improved this on 1/9/15
            inRU = (( errBased_RU(nn).*nVar.*heliVisVar)./( errBased_RU(nn).*nVar + heliVisVar))./nVar;
            if isnan(inRU)
                inRU=0;
            end
            errBased_RU(nn)=inRU;  
        end

        
        if ~isfinite(errBased_RU(nn))
            keyboard
        end
        
    end
    
    % Compute error based CPP for subject data:
    totUnc=(noise(nn).^2)./(1-errBased_RU(nn));
    % note, this should just be the denominator from above. Why didn't I
    % just use that? Not sure.
    
    
    pSame=(1-H).*normpdf(PE(nn), 0, totUnc.^.5).^lw;
    pNew = H .* (1./outcomeSpace).^lw;
    errBased_pCha(nn) = pNew./ (pSame+pNew);
   
    
    if ~isOdd(nn)
        errBased_LR(nn)=errBased_RU(nn)+errBased_pCha(nn)- errBased_RU(nn).*errBased_pCha(nn);
    else
        errBased_LR(nn)=errBased_RU(nn)- errBased_RU(nn).*errBased_pCha(nn);
    end
    
end


% compute model update
errBased_UP=errBased_LR.*PE;
