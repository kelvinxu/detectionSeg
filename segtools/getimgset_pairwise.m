function [ pairwise_table ] = getimgset_pairwise( img, existing_table,verbose )
%GETIMGSET_PAIRWISE 
% [ pairwise_table ] = getimgset_pairwise( img )
% 
% Returns a map.Container object with smooth pairwise potentials for the image
% set
% 
% img = cell of images
% optional: verbose (outputs extra info if set to 1)

%int2str
%str2num

if nargin > 2
   pairwise_table = existing_table; 
else
    pairwise_table = containers.Map;
end

imnum = length(img);
for i = 1: imnum
    [sx,sy,~] = size(img{i});
    key = dim2key(sx,sy);
    if ~pairwise_table.isKey(key)
        pairwise_table(key) = smooth_pairwise([sx,sy],1,1);
    end
    if nargin == 3
        fprintf('Processed %d images...\n', i);
    end
end
end

function [sx,sy] = key2dim(key)
    dims = strsplit(key,'x');
    sx = dims(1);
    sy = dims(2);
end
