function model = mlp2layer_train(data, target, numhid, learnrate, momentum, ...
    weightcost, maxit, displaygap, dataval, targetval, savegap, outdir)
% MLP2LAYER_TRAIN
%
% Train a 2-layer neural network, output layer is  logistic regression.
%
% Usage:
%
% model = mlp2layer_train(data, target, numhid, learnrate, momentum, ...
%    weightcost, maxit, displaygap, dataval, targetval, savegap, outdir)
%
% data: N*D matrix, data(n, :) is the feature vector for the nth data case.
%       D is the dimensionality. This data matrix is padded with an extra
%       dimension of 1's for the bias term.
% target: N*1 vector, target(n) \in [1, K] is the class label for 
%       data(n, :). K is the number of classes.
% numhid: number of hidden units in the first layer denoted as J for 
%       reference.
% model: the network weights, contains two parts, model.w1 and model.w2, 
%       w1 is a J*(D+1) matrix, w2 is a K*(J+1) matrix
%
% [Optional parameters]
% learnrate: learning rate (default: 0.01)
% momentum: 0 if not using momentum (default: 0)
% weightcost: 0 if not using weight-decay (default: 0)
% maxit: maximum number of iterations (default: 100)
% displaygap: number of epochs between two displays (default:1)
% dataval: validation data used to evaluate the performance if provided
% targetval: validation data target
% savegap: number of epochs between two saves, if specified
% outdir: the directory to save models, if specified
%
% NOTE: if it is not working, you may need to play with the learning rate,
% momentum, weightcost and initial network weights (need to change source
% code).
% 
% Yujia Li, 01/2012
%

ismomentum = 1;
isweightdecay = 1;
isval = 0;
issave = 0;

if nargin < 4
    epsilon = 0.1;
else
    epsilon = learnrate;
end
if nargin < 5
%     ismomentum = 0;
      momentum = 0.9;
end
if nargin < 6
    isweightdecay = 0;
end
if nargin < 7
    maxit = 1000;
end
if nargin < 8
    displaygap = 10;
end
if displaygap < 1
    fpringf(stderr, 'Displaygap cannot be smaller than 1!\n');
end
if nargin >= 10
    isval = 1;
end
if nargin >= 12
    if ~isdir(outdir)
        mkdir(outdir)
    end
    issave = 1;
end

[numcases, numdims] = size(data);

K = max(target);
w2 = 1e-4 * randn(K, numhid + 1);
w1 = 1e-4 * randn(numhid, numdims + 1);
w2inc = zeros(K, numhid + 1);
w1inc = zeros(numhid, numdims + 1);

% pad x0 with an extra column of ones
x0 = [ones(numcases, 1), data];
x1pad = ones(numcases, numhid + 1);

t = full(sparse(1:numcases, target, ones(1, numcases), numcases, K));

if isval
    numvalcases = size(dataval, 1);
    xval = [ones(numvalcases, 1), dataval];
    errlog = zeros(3, maxit);
else
    errlog = zeros(2, maxit);
end

fprintf('Training started...\n');
% fprintf([...
%     'Number of Hidden Units: %d\n' ...
%     'Learning Rate: %f\n' ...
%     'Momentum: %f\n' ...
%     'Weightcost: %f\n' ...
%     'Maxit: %d\n'] ...
%     , numhid, learnrate, momentum, weightcost, maxit);

it = 0;
model.w1 = w1;
model.w2 = w2;

tpred = mlp2layer_classify(x0, model, 0);
if isval
    tpredval = mlp2layer_classify(xval, model, 0);
    fprintf(...
        'Epoch %5d. Accuracy: %f  ValAcc: %f  w2inc: %f  w1inc: %f  etime: %f\n', ...
        0, (sum(target == tpred) / numcases), ...
        (sum(targetval == tpredval) / numvalcases), ...
        max(max(abs(w2inc))), max(max(abs(w1inc))), 0);
    tic;
else
    fprintf(...
        'Epoch %5d. Accuracy: %f  w1inc: %f  w2inc: %f  etime: %f\n', ...
        0, (sum(target == tpred) / numcases), max(max(abs(w1inc))), max(max(abs(w2inc))), 0);
    tic;
end


for it = 1 : maxit
    % feed forward
    x1 = sigmoid(x0, w1);
    x1pad(:, 2:numhid + 1) = x1;
    y = softmax_dist_eval(x1pad, w2);
    
    % backprop
    
    w2inct = (t - y)' * x1pad / numcases;
    dldx1 = (t - y) * w2;
    w1inct = (dldx1(:, 2:numhid+1) .* (x1 .* (1 - x1)))' * x0 / numcases;
    
    if ismomentum
        w1inc = momentum * w1inc + epsilon * w1inct;
        w2inc = momentum * w2inc + epsilon * w2inct;
    else
        w1inc = epsilon * w1inct;
        w2inc = epsilon * w2inct;
    end
    
    if isweightdecay
        w2inc = w2inc - epsilon * weightcost * w2;
        w1inc = w1inc - epsilon * weightcost * w1;
    end
    
    w2 = w2 + w2inc;
    w1 = w1 + w1inc;
    
    w2(w2 > 5) = 5;
    w2(w2 < -5) = -5;
    w1(w1 > 5) = 5;
    w1(w1 < -5) = -5;
    
    model.w2 = w2;
    model.w1 = w1;
    
    %{
    if mod(it, displaygap) == 0
        etime = toc;
        tpred = mlp2layer_classify(x0, model, 0);
        fprintf(['%4d iterations passed. Accuracy: %f ' ...
            'w2inc: = %f,  w1inc: = %f,  etime: %f\n'], ...
            it, (sum(target == tpred) / numcases), ...
            max(max(abs(w2inc))), max(max(abs(w1inc))), etime);
        tic;
    end
    %}
    if mod(it, displaygap) == 0
        etime = toc;
        tpred = mlp2layer_classify(x0, model, 0);
        errlog(1,it/displaygap) = it;
        errlog(2,it/displaygap) = sum(target == tpred) / numcases;
        if isval
            tpredval = mlp2layer_classify(xval, model, 0);
            errlog(3,it/displaygap) = sum(targetval == tpredval) / numvalcases;
            fprintf(...
                'Epoch %5d. Accuracy: %f  ValAcc: %f  w2inc: %f  w1inc: %f  etime: %f\n', ...
                it, (sum(target == tpred) / numcases), ...
                (sum(targetval == tpredval) / numvalcases), ...
                max(max(abs(w2inc))), max(max(abs(w1inc))), etime);
            tic;
        else
            fprintf(...
                'Epoch %5d. Accuracy: %f  w1inc: %f  w2inc: %f  etime: %f\n', ...
                it, (sum(target == tpred) / numcases), max(max(abs(w1inc))), max(max(abs(w2inc))), 0);
            tic;
        end
        
    end
    if issave && mod(it, savegap) == 0
        save([outdir '/m' num2str(it)], 'model');
    end
    %{
    if mod(it, displaygap) == 0
        etime = toc;
        tpred = mlp2layer_classify(x0, model, 0);
        fprintf('%4d iterations passed. Accuracy: %f   timecost: %f\n', ...
            it, (sum(target == tpred) / numcases), etime);
        tic;
    end
    %}
end

errlog = errlog(:, 1:ceil(maxit/displaygap));

if issave
    save([outdir '/errlog'], 'errlog');
end

return
end
