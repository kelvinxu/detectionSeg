% load('/datasets/features.mat');
% load('/datasets/groundtruth.mat');
% unary = get_unary(normalized_feats, model, featnum);
% p = smooth_pairwise_uniform([100,100],1);
arr_range = cell(2,1);
arr_range{2}  =[1:15];
arr_range{3} = [0.1:0.5:0.5];
acclog = tune_crf(unary, {}, n_im, arr_range, groundtruth);