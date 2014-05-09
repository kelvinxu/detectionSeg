function [predictions] = predict_boykov(unary, img, sigma, relative_weights)
%Predict

predictions = {};
imnum = length(unary);
bk_pairwise = boykov_pairwise(img, sigma);

for i = 1 : imnum
    predictions{i} = pixlabel_inf_crf(unary{i}, bk_pairwise{i}, 1 - eye(2), relative_weights); 
end
end

