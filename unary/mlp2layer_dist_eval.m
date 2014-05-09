function y = mlp2layer_dist_eval(x, model, ispad, t)
% MLP2LAYER_DIST_EVAL
%
% Evaluate softmax distribution of the output layer in the 2-layer MLP.
%
% Usage:
% y = mlp2layer_dist_eval(x, model, ispad)
% y = mlp2layer_dist_eval(x, model, ispad, t)
%
% x: N*(D+1) data matrix, the first dimension of each datacase is constant 1
% model: contains w1 and w2, w1 is a J*(D+1) matrix, w2 is a K*(J+1) matrix
% ispad: whether to padd x or not
% t: N*1 label matrix, if specified, only evaluate p(y_i=t_i|x_i, w) for
% each i.
%
% Yujia Li, 01/2012
%

numcases = size(x, 1);

if nargin >= 3 && ispad
    x0 = [ones(numcases, 1), x];
else
    x0 = x;
end

w1 = model.w1;
w2 = model.w2;

x1 = [ones(numcases, 1), sigmoid(x0, w1)];
if nargin > 3
    y = softmax_dist_eval(x1, w2, 0, t);    % output is a softmax layer
else
    y = softmax_dist_eval(x1, w2, 0);
end

return
end