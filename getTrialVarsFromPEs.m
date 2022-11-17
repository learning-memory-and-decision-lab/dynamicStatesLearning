function [errBased_pCha, errBased_RU, errBased_LR, errBased_UP]=getTrialVarsFromPEs...
    (noise, PE, modHaz, separateCPP_RU, relUnc, newBlock, allHeliVis,...
    initRU, heliVisVar, lw, ud)


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


if separateCPP_RU
    
    % PART 1: Create CPP estimate that incorporates subjective error
    % information.
    
    allNoise=unique(noise);
    avRelUnc=nanmedian(relUnc);
    
    for i = 1:length(allNoise)
        % Compute error based CPP for subject data:
        sel=noise==allNoise(i);
        totUnc=(allNoise(i).^2)./(1-avRelUnc);
        pSame=(1-H).*normpdf(PE(sel), 0, totUnc.^.5).^lw;
        pNew = H .* (1./300).^lw;
        errBased_pCha(sel) = pNew./ (pSame+pNew);
    end
    
    % PART 2: get relative uncertainty estimates that are based on subjective error
    % information.
    % NOTE: This requires some assumptions about initial uncertainty and catch
    % trial driven changes in uncertainty. Here we will assume that subjects
    % become completely certain after catch trials and start the task quite
    % uncertain... but these assumptions will be softened later in the analyis
    
    
    for i = 1:length(errBased_pCha)
        if newBlock(i)
            errBased_RU(i)=initRU;
        else
            nVar=noise(i-1).^2;
            cp   =errBased_pCha(i-1);
            tPE  =PE(i-1);
            
            inRU=errBased_RU(i-1);
             
            
            % from equation 6 in heliFMRI paper:
            numerator=(cp.*nVar)+((1-cp).*inRU.*nVar) + ...
                cp.*(1-cp).*(tPE.*(1-inRU)).^2;
            numerator=numerator./ud;
            denominator = numerator+nVar;
            errBased_RU(i)=numerator./denominator;
            
                     
            % if you can see the helicopter, update uncertainty accordingly.
            if allHeliVis(i)==1
                
                inRU= ((errBased_RU(i).*nVar.*heliVisVar)./(errBased_RU(i).*nVar + heliVisVar))./nVar;
                if isnan(inRU)
                    inRU=0;
                end
                errBased_RU(i)=inRU;
            end
  
            
            
            
            if ~isfinite(errBased_RU(i))
                keyboard
            end
            
        end
    end

else
    
    
% Compute both CPP and RU according to subject prediction errors, not just data:    
    
    
    for i = 1:length(noise)
        
        if newBlock(i)
            errBased_RU(i)=initRU;
        else
            nVar=noise(i-1).^2;
            cp   =errBased_pCha(i-1);
            tPE  =PE(i-1);
            inRU=errBased_RU(i-1);
            
            % from equation 6 in heliFMRI paper:
            numerator=(cp.*nVar)+((1-cp).*inRU.*nVar) + ...
                cp.*(1-cp).*(tPE.*(1-inRU)).^2;
            
            numerator=numerator./ud; % divide uncertainty about mean by constant
            denominator = numerator+nVar; % denominator is just numerator plus noise variacne
            errBased_RU(i)=numerator./denominator; % RU is just the fraction
            
                        
            if allHeliVis(i)==1
             % MRN improved this on 1/9/15
              inRU = (( errBased_RU(i).*nVar.*heliVisVar)./( errBased_RU(i).*nVar + heliVisVar))./nVar;
                if isnan(inRU)
                    inRU=0;
                end
                errBased_RU(i)=inRU; 
                
            end


            if ~isfinite(errBased_RU(i))
                keyboard
            end
            
        end
        
        % Compute error based CPP for subject data:
        totUnc=(noise(i).^2)./(1-errBased_RU(i));
        pSame=(1-H).*normpdf(PE(i), 0, totUnc.^.5).^lw;
        pNew = H .* (1./300).^lw;
        errBased_pCha(i) = pNew./ (pSame+pNew);
       
    end
    
    
    % compute model update
    errBased_LR=errBased_RU+errBased_pCha- errBased_RU.*errBased_pCha;
    errBased_UP=errBased_LR.*PE;
    
    
end