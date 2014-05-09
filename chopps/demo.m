% This demo shows how to use the Compositional High Order Patten Potentials
% to do inference and also how to learn them.
%
% See the following paper for more technical details:
% - Yujia Li, Daniel Tarlow and Richard Zemel. Exploring Compositional High 
%   Order Pattern Potentials for Structured Output Learning. The 26th IEEE 
%   Conference on Computer Vision and Pattern Recognition (CVPR 2013).
%
% Yujia Li, 09/2013
%

% this adds the graph cut optimization package to the path.  The gco
% package is a copy of gco-v3.0 available from 
% http://vision.csd.uwo.ca/code/
addpath('gco/matlab');

% load sample data, the weizmann horse data set and pretrained unary and
% pairwise potentials.
load data/weizmann_32_32_trainval.mat

% visualize training segmentations
fprintf('Visualizing the training data.\n')
figure, patchviewx(xtrain, [32,32], 10, 0, 0, 0, 1);
input('');

% make predictions on training data based on unary potential only
y = pixlabel_inf_unary(utrain);

% visualize predictions
fprintf('Visualizing the predictions of the unary model.\n')
figure, patchviewx(y, [32,32], 10, 0, 0, 0, 1);
input('');

% compute intersection over union
iou = mean(sum(y & xtrain, 2) ./ sum(y | xtrain, 2))
input('');

% make predictions using both unary potentials and pairwise potentials
y = pixlabel_inf_crf(utrain, ptrain, 1 - eye(2), 1);

fprintf('Visualizing the predictions of the unary+pairwise model.\n')
figure, patchviewx(y, [32,32], 10, 0, 0, 0, 1);
input('');
iou = mean(sum(y & xtrain, 2) ./ sum(y | xtrain, 2))
input('');

% train the RBM model (independently trained RBM)
fprintf('Learning the RBM model.\n')
brbm_pcd_sparse(xtrain, 32, 1, 1e-1, 0, 0, 0.1, 1e-1, 40, 100, 1000, 0, 20, 'temp/rbm', 100);

% load the learned RBM model
load temp/rbm/m1000.mat

% visualize the learned compositional patterns
fprintf('Visualizing learned compositional patterns.\n')
figure, patchviewx(vishid', [32,32], 4, 0, 0, 0, 1);
input('');

% find the best lambda, which is the relative importance for the CRF part of 
% the model (the weight for the RBM part is fixed to be one). This is to keep
% the two parts on the same scale.
[acclog, lambda] = tune_lambda_p(utrain, ptrain, vishid, visbiases, hidbiases, 1, 0:0.1:1, xtrain);

% make predictions with the pretrained RBM and the CRF (unary+pairwise), 
% i.e. the CHOPPs model, using the EM inference algorithm.
y = pixlabel_inf_em_p(utrain, ptrain, vishid, visbiases, hidbiases, 50, lambda, 1);

fprintf('Visualizing predictions of CHOPPs model.\n')
figure, patchviewx(y, [32,32], 10, 0, 0, 0, 1);
input('');
iou = mean(sum(y & xtrain, 2) ./ sum(y | xtrain, 2))
input('');

% make a copy of the RBM model, required to call the function cbrbm_mel_init_p
w = make_w(vishid, visbiases, hidbiases);   

% Jointly learn RBM parameters by minimizing expected loss
fprintf('Joinly learn CHOPPs parameters by minimizing expected loss.\n')
cbrbm_mel_init_p(xtrain, utrain, ptrain, lambda, 1, w, 1, 1e1, 0, 0, 0.1, 0, 40, 2, 500, 0, 10, 0, 1, 0, 'temp/crbm', 10, xval, uval, pval);

% load the learned model
load temp/crbm/m500.mat

% make predictions with the jointly learned CHOPPs model.
y = pixlabel_inf_em_p(utrain, ptrain, vishid, visbiases, hidbiases, 50, lambda, 1);

fprintf('Visualizing predictions of the joinly trained CHOPPs model.\n')
figure, patchviewx(y, [32,32], 10, 0, 0, 0, 1);
input('');
iou = mean(sum(y & xtrain, 2) ./ sum(y | xtrain, 2))
input('');

