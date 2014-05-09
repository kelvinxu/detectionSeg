function [t, top] = mlp2layer_classify(data, model, ispad, topk)
% MLP2LAYER_CLASSIFY
%
% Classify test cases using multi-layer perceptron
%
% [t, top] = mlp2layer_classify(data, model, ispad, topk)
%
% data: N*D matrix, each row is a feature vector
% model: contains w1 and w2, w1 is of size J*(D+1), w2 K*(J+1)
% ispad: if true, data is padded with one extra column of ones, otherwise,
% data is treated to have dimensionality D+1. Default: 0.
%
% t: a column vector, t(n) is the class label for the nth data point.
%
% Yujia Li, 01/2012
%

numcases = size(data, 1);

if nargin == 3 && ispad
    x0 = [ones(numcases, 1), data];
else
    x0 = data;
end

w1 = model.w1;
w2 = model.w2;

x1 = [ones(numcases, 1), sigmoid(x0, w1)];
if nargout < 2
    t = softmax_regression_classify(x1, w2, 0);
else
    [t, top] = softmax_regression_classify(x1, w2, 0, topk);    % output is a softmax layer
end

return
end
