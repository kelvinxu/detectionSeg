function featurelist = filter_response(imlist, bndboxes, isnoise)
% FILTER_RESPONSE
%
% Usage:
% featurelist = filter_response(imlist)
%
% imlist is a cell array of images.
% 
% featurelist is a cell array of feature matrices, each element of this array
% will be a N*D matrix, N is the number of pixels in that image. D is the
% number of features extracted for that pixel.
%
% Yujia Li, 04/2012
%
% if bndboxes included, we extract two extra features

extra_feats = 6;

numcases = length(imlist);
featurelist = cell(numcases, 1);

cform = makecform('srgb2lab');
% Gabor filter depends on the size of the image
% gbfilter = sg_createfilterbank(size(dbim), 0.2, 5, 4);
lmfilter = makeLMfilters();
sfilter = makeSfilters();


for i = 1 : numcases
    if nargin > 1
    featurelist{i} = applyfilterbank(imlist{i}, cform, lmfilter, sfilter, extra_feats, bndboxes{i}, isnoise);
    else 
    featurelist{i} = applyfilterbank(imlist{i}, cform, lmfilter, sfilter);
    end
    
    if mod(i, 1) == 0
        fprintf('Processed %d images...\n', i);
    end
end

return
end
