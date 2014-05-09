function unary = get_unary(imhov, model, featurenum)
% GET_UNARY
%
% Get unary potentials used in a CRF, the model can be either a softmax
% regression or a 2 layer neural net.
%
% Usage:
% unary = get_unary(imhov, model)
%
% imhov can be either the feature matrix for a single image or a cell array
% containing features for a list of images. Featurenum allows for you to 
% select a feature from 108-113 for inclusion.
%
% Yujia Li, 04/2012
%

if nargin > 2 && iscell(imhov)
    features =[1:107, 107 + featurenum];
end

if iscell(imhov)
    numcases = length(imhov);
    unary = cell(numcases, 1);

    for i = 1 : numcases
        unary{i} = get_unary_single(imhov{i}(:, features), model);
    end
else
    unary = get_unary_single(imhov, model);
end

return
end

function unary = get_unary_single(imhov, model)

if isstruct(model)
    unary = mlp2layer_dist_eval(imhov, model, 1);
else
    unary = softmax_dist_eval(imhov, model, 1);
end

% unary potentials should be of size [NumLabels * NumSites]
unary = unary';

unary = -log(unary);


return
end

