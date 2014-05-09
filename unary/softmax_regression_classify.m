function [t, top] = softmax_regression_classify(x, w, ispad, topk)
% SOFTMAX_REGRESSION_CLASSIFY
%
% Use softmax regression to classify test data, i.e. do prediction.
%
% Usage:
% [t, top] = softmax_regression_classify(x, w, ispad, topk)
%
% x: N*D or N*(D+1) data matrix, use ispad parameter to specify whether
%     padding is needed.
% w: K*(D+1) weight matrix.
% ispad: if used, pad x with an extra dimension of 1's for the bias term.
% topk: if set, return topk results
%
% Output:
% t: N*1 class labels
% [t, top]: top is a N*topk matrix, each row is the topk result for a data case
% 
% Yujia Li, 01/2012
%
y = softmax_dist_eval(x, w, ispad);

if nargout < 2
    [p, t] = max(y, [], 2);
else
    [p, tidx] = sort(y, 2, 'descend');
    t = tidx(:,1);
    top = tidx(:,1:topk);
end

return
end
