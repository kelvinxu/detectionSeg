function [predictions, pairwise_table] = predict_pairwise(unary, pairwise_table, img, relative_weights)
%Predict

predictions = {};

imnum = length(unary);

for i = 1 : imnum
    [sx,sy, ~] = size(img{i});
    key = dim2key(sx,sy);
    if pairwise_table.isKey(key)
        p = pairwise_table(key);
    else
        p = smooth_pairwise([sx,sy],1,1);
        pairwise_table(key) = p;
    end
    predictions{i} = reshape(pixlabel_inf_crf(unary{i}, p{1}, 1 - eye(2), relative_weights),[sx,sy]); 
%     figure, imshow(predictions{i});
end
end

