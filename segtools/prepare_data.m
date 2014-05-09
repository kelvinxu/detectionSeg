function [tdata, tlabels, vdata, vlabels, mean, std] = prepare_data(img, labels, split, bndboxes, isnoise)
% Takes in images, and splits the sets as determined by split, 
% then extracts some features to be trained by a later model. 
% 
% This function takes an biased sample to ensure a 50-50 split. 
% 
% [tdata, tlabels, vdata, vlabels, mean, std] = prepare_data(img, labels, split, bndboxes, isnoise)
% 
% Usage: 
% img should be a cell of N images
% labels: should be a cell N label matrices
% split: this is the proportion of training images (0 <=split <=1)
% 
% Kelvin Xu 02/27/14

% we sample 10% of pixels per image
sample_size = 0.25;
nfeats = 107;

if nargin >3
   % add six bounding box features
   nfeats = 107+6;
end

%randomize the image's selected
 tindex = floor(length(img)*split);
 indx = randperm(length(img));
 img = img(indx);
 labels = labels(indx);
 bndboxes = bndboxes(indx);
if nargin > 3
%     bndboxes = bndboxes(indx);
    t_feats = filter_response(img(1:tindex), bndboxes(1:tindex),isnoise);
    v_feats = filter_response(img(tindex+1:length(img)),bndboxes(tindex+1:length(img)),isnoise);
else
    t_feats = filter_response(img(1:tindex+1));
    v_feats = filter_response(img(tindex+1:length(img)));
end


%get total image dimensions, this is an overestimation that will be trimmed
%later

t_size = total_pixels(t_feats, sample_size);
v_size = total_pixels(v_feats, sample_size);

[tdata, tlabels] = down_sample(t_feats, labels(1:tindex),t_size, sample_size,nfeats);
[vdata, vlabels] = down_sample(v_feats, labels(tindex+1:length(labels)), v_size, sample_size,nfeats);

% 1 indicates an airplane
% sets all the necessary bits
% Yujia's nn code needs doubles

tlabels = set_bits(tlabels);
vlabels = set_bits(vlabels);

[tdata, mean, std]= normalize_data(tdata);
[vdata,~ ,~] = normalize_data(vdata, mean, std);

end
%%
%    Helper functions 

function [data,labels] = down_sample(features, flabels, total_size, sample_size,nfeats)

data = zeros([total_size,nfeats]);
labels = zeros([total_size, 1]);

current_feat = 0;
for i=1:length(features)
%   Get length of response and randomly pick a few
    [npix,~] = size(features{i});
    nsampled = round(npix*sample_size);
    flat_labels = reshape(flabels{i},[npix,1]);
    
    % do a roughly fifty-fifty split on positive/negative examples
    positive_examples = find(flat_labels == 1);
    % pixels in bndbox that are not part of the image 
    bndbox_pixels = setdiff(find(features{i}(:, 108) ~= 0), positive_examples);
    % negative examples not in the image
    negative_examples = setdiff(find(flat_labels ~= 1), bndbox_pixels);

%% Work in Progress: How should we sample?
    % depending on the makeup of the image, we sample in a different way
    % I want to sample something like 0.5 
    length_array = [length(positive_examples), length(bndbox_pixels), length(negative_examples)];
    [smallest_size ,smallest_set] = min(length_array);
    
    switch smallest_set
        case 1
            psize = min([smallest_size, nsampled*0.5]);
            bsize = psize*0.2;
            nsize = psize*0.8;
        case 2
            bsize = min([smallest_size, round(nsampled*0.25)]);
            psize = bsize*2;
            nsize = bsize;
        case 3
            nsize = min([smallest_size, round(nsampled*0.25)]);
            psize = nsize*2;
            bsize = nsize;
    end
    
    % this is meant to catch any weird corner cases. 
    psize = min([length_array(1), psize]);
    bsize = min([length_array(2), bsize]);
    nsize = min([length_array(3), nsize]);
    
    rand_pixels = [positive_examples(randperm(floor(psize))); ... 
        negative_examples(randperm(floor(nsize)));
        bndbox_pixels(randperm(floor(bsize)))];
%%
    
%   Take the corresponding data points and labels
    data(current_feat+1:current_feat+length(rand_pixels),:) = features{i}(rand_pixels,:);
    labels(current_feat+1:current_feat+length(rand_pixels),1) = double(flat_labels(rand_pixels,:));
    current_feat = current_feat + length(rand_pixels);
end

%this returns the first row of pure zero
ix = find(any(sum(data,2) == 0,2),1);

data(ix:length(data),:) = [];
labels(ix:length(labels)) = [];

end

function totalsize = total_pixels(feats,sample_size)
% feats is a cell of features matrices
% get total size of pixels
totalsize = 0;
for i=1:length(feats)
    [npix,~] = size(feats{i});
    totalsize = totalsize + round(npix*sample_size);
end 
end
