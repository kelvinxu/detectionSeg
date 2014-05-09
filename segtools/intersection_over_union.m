function [pixacc, iouacc] = intersection_over_union(labeling, ground_truth)
%
%   Put header here
%
%
%

imnum = length(labeling);

numcorrect_im = zeros(imnum, 1);
pixel_num = 0;
iouacc_im = zeros(imnum, 1);

% compute pixel accuracy 
for i = 1 : imnum
    numcorrect_im(i) = sum(labeling{i} == ground_truth{i})
    pixel_num = pixel_num +  prod(labeling{i});
end

pixacc = sum(numcorrect_im)/pixel_num;

% compute iou accuracy
for i = 1 : imnum
    iouacc_im(i) = sum(labeling{i} & ground_truth{i}) / sum(labeling{i} | ground_truth{i}); 
end

iouacc = mean(iouacc);
end

function score = intersection_over_union_single(a,b, label)
% computes standard intersection over union metric
% Kelvin

a = (a ==label);
b = (b == label);
score = sum((a(:) & b(:)))/sum(a(:) | b(:));

end

