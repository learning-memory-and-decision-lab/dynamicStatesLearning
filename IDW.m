function[interpAct] = IDW(currentValue,outputAct)
outputMeans = -50:10:350;% "Output Layer" with Gaussian units
[~,idx]=sort(abs(outputMeans-currentValue));
NN = idx(1:2);
Distance = abs((currentValue - outputMeans(NN)));
wV = (10-Distance) .* outputAct(NN);
interpAct = sum(wV)./10;
    
    