function brbm_pcd_sparse(data, numhids, nCD, learnrate, momentum, weightcost, ...
    sparsetarget, sparseweight, batchsz, numsamples, maxepoch, decgap, ...
    displaygap, outdir, savegap, seed)
% BRBM_PCD_SPARSE
%
% This is an implementation of the most fundamental binary RBM (binary 
% visibles and hiddens) using PCD.
%
% Usage:
% brbm_pcd_sparse(data, numhids, nCD, learnrate, momentum, weightcost, ...
%    sparsetarget, sparseweight, batchsz, numsamples, maxepoch, decgap, ...
%    displaygap[, outdir, savegap, seed])
%
% --- Inputs ---
% data: N*D matrix, each row is a data case
% numhids: number of hidden variables
% nCD: number of CD runs, if set, the number of CD runs will be fixed to that
%       value.
% learnrate: learning rate
% momentum: momentum
% weightcost: parameter for weightdecay
% sparsetarget: target average hidden activity
% sparseweight: the weight for the sparse penalty
% batchsz: size of a minibatch
% numsamples: number of persistent samples
% maxepoch: number of epochs to run
% decgap: if set to 0, a constant learning rate will be used through out the
%       learning procedure. Otherwise the learning rate will decrease every
%       decgap epochs
% displaygap: number of epochs between two displays
% outdir: the directory to store models, will be created if not exist
% savegap: number of epochs between two saves.
% seed: seed used for random number generator, if set.
%
% --- Output ---
% Model parameters vishid, visbiases, hidbiases will be stored in model files
% in outdir.
%
% Yujia Li, 03/2012
%

isoutput = 1;

if nargin < 15
    isoutput = 0;
else
    if ~isdir(outdir)
        mkdir(outdir);
    end
end

if nargin < 16
    rand('seed',sum(100*clock));
    randn('seed',sum(100*clock));
else
    rand('seed', seed.rand);
    rand('seed', seed.randn);
end

warning off;

epsilonw  = learnrate;
epsilonvb = learnrate;
epsilonhb = learnrate;

[numtotalcases, numdims] = size(data);
numbatches = floor(numtotalcases / batchsz);

vishid = epsilonw * randn(numdims, numhids);
hidbiases = zeros(1, numhids);
visbiases = zeros(1, numdims);

vishidinc  = zeros(numdims, numhids);
hidbiasinc = zeros(1, numhids);
visbiasinc = zeros(1, numdims);

% samples kept in the persistent chain
visk = zeros(numsamples, numdims);
hidk = zeros(numsamples, numhids);

errlog = zeros(1, maxepoch);

fprintf(['Training starts...\n' ...
    'numhids: %d\n' ...
    'learnrate: %f\n' ...
    'nCD: %d\n' ...
    'momentum: %f\n' ...
    'weightcost: %f\n' ...
    'maxepoch: %d\n' ...
    ], numhids, learnrate, nCD, momentum, weightcost, maxepoch);

tic;

for epoch = 1 : maxepoch
    errsum = 0;

    % randomly reorder all the data cases
    idx = randperm(numtotalcases);
    % idx = 1:numtotalcases;

    if decgap > 0 && mod(epoch, decgap) == 0
        epsilonw  = learnrate / (epoch / decgap + 1);
        epsilonvb = learnrate / (epoch / decgap + 1);
        epsilonhb = learnrate / (epoch / decgap + 1);
    end

    for batch = 1 : numbatches
        istart = (batch - 1) * batchsz + 1;
        if batch ~= numbatches
            iend = istart + batchsz - 1;
        else
            iend = numtotalcases;
        end

        x = data(idx(istart:iend), :);

        [numcases, numdims] = size(x);
        
        % sample h ~ h|x
        poshidprobs = 1 ./ (1 + exp(-x * vishid - ...
            ones(numcases, 1) * hidbiases));
        hid0 = poshidprobs > rand(numcases, numhids);
        % hidk = hid0;
 
        posprods = x' * poshidprobs;
        poshidact = sum(poshidprobs);
        posvisact = sum(x);

        % reconstruction of visible units, used in measuring reconstruction error
        recvisprobs = 1 ./ (1 + exp(-hid0 * vishid' - ...
            ones(numcases, 1) * visbiases));
        visrec = recvisprobs > rand(numcases, numdims);
        
        % visk = zeros(numcases, numdims);
 
        for cd = 1 : nCD
            negvisprobs = 1 ./ (1 + exp(-hidk * vishid' - ...
                ones(numsamples, 1) * visbiases));
            visk = negvisprobs > rand(numsamples, numdims);
 
            neghidprobs = 1 ./ (1 + exp(-visk * vishid - ...
                ones(numsamples, 1) * hidbiases));
            hidk = neghidprobs > rand(numsamples, numhids);
        end
 
        % CD finished, calculate error signals
        
        negprods = visk' * neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(visk);
        
        vishidinct = epsilonw * (posprods / numcases - negprods / numsamples);
        visbiasinct = epsilonvb * (posvisact / numcases - negvisact / numsamples);
        hidbiasinct = epsilonhb * (poshidact / numcases - neghidact / numsamples);
        
        vishidinc = momentum * vishidinc + vishidinct;
        visbiasinc = momentum * visbiasinc + visbiasinct;
        hidbiasinc = momentum * hidbiasinc + hidbiasinct;
        
        vishidinc = vishidinc - epsilonw * weightcost * vishid;
        
        % sparse constraints
        if batch == 1
            hidactold = sparsetarget * ones(1, numhids);
        else
            hidactold = 0.9 * hidactold + 0.1 * neghidact / numsamples;
        end
        vishidinc = vishidinc - epsilonw * sparseweight * negvisact' / numsamples * (hidactold - sparsetarget);
        hidbiasinc = hidbiasinc - epsilonhb * sparseweight * (hidactold - sparsetarget);

        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
 
        vishid(vishid > 5) = 5;
        vishid(vishid < -5) = -5;
        visbiases(visbiases > 8) = 8;
        visbiases(visbiases < -8) = -8;
 
        % err = sum(sum((x - visk).^2));
        err = sum(sum((x - visrec).^2));
        errsum = errsum + err;
    end

    if mod(epoch, displaygap) == 0
        fprintf(['epoch %d, err %.2f, vishidinc %.4f, visbiasinc %.4f, ' ...
            'hidbiasinc %.4f, hidact %.4f, time %.2f\n'], epoch, errsum, ...
            max(max(abs(vishidinc))), max(abs(visbiasinc)), ...
            max(abs(hidbiasinc)), mean(hidactold), toc);
        tic;
    end
    if isoutput && mod(epoch, savegap) == 0 
        save([outdir '/m' ...
            num2str(epoch) '.mat'], 'vishid', 'hidbiases', 'visbiases');
    end
    
    errlog(epoch) = errsum;
end

if isoutput
    save([outdir '/model_err.mat'], 'errlog');
end

return
end

