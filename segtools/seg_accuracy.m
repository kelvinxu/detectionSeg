function [ score ] = seg_accuracy( pred, labels )
%seg_accuracy.m: computes the overall accuracy of segmentation
%  
% Usage: 
% 
% predictions is a cell of images
% labels = a cell of corresponding labels containing a field cdata
% 
% score is the average percentage, computed as the intersection over union
% the second dimension is the pixel accuracy

score = zeros(1, length(pred));

for i=1:length(pred)
    [sx,sy] = size(pred{i});
    score(i) = intersection_over_union_single(~reshape(pred{i}{1},100,100), set_bits(labels{i}));
%     score(2,i) = pixel_acc(pred{i}+1, set_bits(labels{i}.cdata))/(sx*sy);
end

end

function score = intersection_over_union_single(a,b)
% computes standard intersection over union metric
% Kelvin

score = sum(sum((a & b)))/sum(sum(a | b));

end