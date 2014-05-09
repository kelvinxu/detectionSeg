function [ features ] = select_feature( features_all, featnum)
% Extracts feature feat matrix 

imnum = length(features_all);
features = cell(1,imnum);

for i = 1:imnum
    [sx,sy] = size(features_all{i});
    features{i} = zeros(sx, 108);
    features{i}(:,1:107) = features_all{i}(:,1:107);
    features{i}(:,108) = features_all{i}(:,107+featnum);
end
end


