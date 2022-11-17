
% This is a code for finding the corresponding context shift for different 
% learning rates:

function [s] = LR2shift(allContextIncs, LRs, val)

% Inputs:
% allContextIncs = vector of all context shift increments
% LRs            = vector of all learning rates
% val            =   desired learning rate

% Output:        
% s              = corresponding context shift for "val"

% Find the coefficients.
%coeffs = polyfit(allContextIncs,LRs, 5);

%interpolatedX = linspace(0, 1, 500);
%interpolatedY = polyval(coeffs, interpolatedX);

% Closest value in interpolatedY to desired learning rate:
[~, ix ] = sort( abs( LRs-val ) );
 a = abs(LRs(ix(1)) - val);
 b = abs(LRs(ix(2)) - val);

 % Corresponding Context shift:
 s=((allContextIncs(ix(1)) * a + allContextIncs(ix(2))*b))/(a+b);