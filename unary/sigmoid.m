function y = sigmoid(x, w, ispad)
% SIGMOID
%
% Calculates the logistic sigmoid response to the given input.
%
% Usage:
% y = sigmoid(x, w, ispad)
%
% x: N*D input data array. Each row is a feature vector for one data point.
% w: K*(D+1) weight matrix
% ispad: if it is true, then pad x with one extra column of ones to the
% left. Otherwise treat feature vectors in x as D+1 dimensional. Default: 0
%
% y: N*K output matrix. Each row is the K responses of corresponding
% sigmoid functions.
%
% Yujia Li, 01/2012
%

numcases = size(x, 1);

if nargin == 3 && ispad
    x = [ones(numcases, 1), x];
end

act = x * w';

y = 1 ./ (1 + exp(-act));

return
end