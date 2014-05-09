function y = softmax_dist_eval(x, w, ispad, t)
% SOFTMAX_DIST_EVAL
%
% Evaluate softmax distribution
%
% Usage:
% y = softmax_dist_eval(x, w, ispad)
% y = softmax_dist_eval(x, w, ispad, t)
%
% x: N*(D+1) data matrix, the first dimension of each datacase is constant 1
% w: K*(D+1) weight matrix
% ispad: whether to padd x or not
% t: N*1 label matrix, if specified, only evaluate p(y_i=t_i|x_i, w) for
% each i. If t is just one number, then it is replicated N times.
%
% Yujia Li, 01/2012
%

N = size(x, 1);

if nargin > 2 && ispad
    x = [ones(N, 1), x];
end

K = size(w, 1);

a = x * w';
yact = exp(a - max(a, [], 2) * ones(1, K));
y = yact ./ (sum(yact, 2) * ones(1, K));

if nargin > 3
    step = 0:K:(N-1) * K;
    yp = y';
    y = yp(t + step);
end

return
end
