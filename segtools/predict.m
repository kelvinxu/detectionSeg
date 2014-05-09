function predictions = predict(model, testim, testbndboxes)
%Predict

predictions = {};

if nargin >2
features = filter_response(testim, testbndboxes);
else
features = filter_response(testim);
end

for i=1:length(features)
    [sx,sy,~] = size(testim{i});
    model_output = mlp2layer_classify(normalize_data(features{i}), model, 1);
    predictions{i} = reshape(model_output, [sx,sy]);
end

% Reshape the images after predicting

end

