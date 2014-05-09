function unaries = unary_pairwise(model, testim, testbndboxes)

unaries = {};

if nargin >2
features = filter_response(testim, testbndboxes);
else
features = filter_response(testim);
end

for i=1:length(features)
    [sx,sy,~] = size(testim{i});
    model_output = mlp2layer_dist_eval(normalize_data(features{i}), model, 1);
    unaries{i} = reshape(model_output, [sx,sy,2]);
end
end