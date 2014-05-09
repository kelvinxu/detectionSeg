function cbrbm_mel_init_p(data, unary, pairwise, lambda, mu, initw, nCD, learnrate, ...
    momentum, weightcost, sparsetarget, sparseweight, batchsz, numsamples, maxepoch, decgap, ...
    displaygap, restartgap, isiou, initu, outdir, savegap, xval, uval, pval, seed)
% CBRBM_MEL_INIT_P
%
% This is an implementation of the conditional RBM with preinitialized
% parameter settings.
%
% Usage:
% cbrbm_mel_init_p(data, unary, pairwise, lambda, mu, initw, nCD, learnrate, ...
%    momentum, weightcost, sparsetarget, sparseweight, batchsz, numsamples, maxepoch, decgap, ...
%    displaygap, restartgap, isiou, initu, outdir, savegap, xval, uval, pval, seed)
%
% --- Inputs ---
% data: N*D matrix, each row is a data case
% unary: N-dim cell array, each element is a 2*D matrix representing the unary
%       potentials used in the model
% pairwise: N-dim cell array, each element is a matrix encoding the pairwise 
%       terms for one data case
% lambda: the coefficient for unary potentials, the larger lambda is, the more
%       important unary potentials are in the inference
% mu: coefficient for the ratio of pairwise potentials over unary potentials
% initw: including vh, vb, hb. initial parameter settings
% nCD: number of CD runs, if set, the number of CD runs will be fixed to that
%       value.
% learnrate: learning rate
% momentum: momentum
% weightcost: parameter for weightdecay
% sparsetarget: target average hidden activity
% sparseweight: the weight for the sparse penalty
% batchsz: size of a minibatch
% numsamples: number of samples for each data point, numsamples should be at
%       least 2.
% maxepoch: number of epochs to run
% decgap: if set to 0, a constant learning rate will be used through out the
%       learning procedure. Otherwise the learning rate will decrease every
%       decgap epochs
% displaygap: number of epochs between two displays
% restartgap: number of epochs between restarting the sampling chains
% isiou: If 1, use the iou induced loss, otherwise (0) use per pixel accuracy
%       induced loss.
% initu: If set to 1, initialize the sampling chain from unary predictions, 
%       otherwise use u+p predictions instead.
% outdir: the directory to store models, will be created if not exist
% savegap: number of epochs between two saves.
% xval, uval, pval: if set, the program would also output the performance on
%       validation set during training.
% seed: seed used for random number generator, if set.
%
% --- Output ---
% Model parameters vishid, visbiases, hidbiases will be stored in model files
% in outdir.
%
% Yujia Li, 10/2012
%

isoutput = 1;

if nargin < 22
    isoutput = 0;
else
    if ~isdir(outdir)
        mkdir(outdir);
    end
end

isval = 1;
if nargin < 25
    isval = 0;
end

if nargin < 26
    rand('seed',sum(100*clock));
    randn('seed',sum(100*clock));
else
    rand('seed', seed.rand);
    randn('seed', seed.randn);
end

warning off;

epsilonw  = learnrate;
epsilonvb = learnrate;
epsilonhb = learnrate;

[numtotalcases, numdims] = size(data);
numbatches = floor(numtotalcases / batchsz);
numbatches(numbatches < 1) = 1;

vh = initw.vh;
vb = initw.vb;
hb = initw.hb;

numhids = length(hb);

vishid = vh;
hidbiases = hb;
visbiases = vb;

vishidinc  = zeros(numdims, numhids);
hidbiasinc = zeros(1, numhids);
visbiasinc = zeros(1, numdims);

% samples kept in the persistent chain

numcases = batchsz;
numcases(numcases > numtotalcases) = numtotalcases;

unarymx = zeros(numtotalcases, numdims);
for i = 1 : numtotalcases
    unarymx(i,:) = unary{i}(1,:) - unary{i}(2,:);
end
unarymx = unarymx * lambda;

visk = cell(numsamples, 1);
hidk = cell(numsamples, 1);

lbcost = [0,1;1,0];

for isample = 1 : numsamples
    if initu
        visk{isample} = double(unarymx < 0);
    else
        visk{isample} = pixlabel_inf_crf(unary, pairwise, lbcost, mu);
    end
    hidprobs = 1 ./ (1 + exp(-visk{isample} * vishid - ones(numtotalcases, 1) * hidbiases));
    hidk{isample} = hidprobs > rand(numtotalcases, numhids);
end

clear hidprobs

viskbatch = cell(numsamples, 1);
hidkbatch = cell(numsamples, 1);
hidprobsbatch = cell(numsamples, 1);

% errlog = zeros(maxepoch, 2);
errlog = zeros(maxepoch, 1);
if isval
    acclog = zeros(floor(maxepoch / displaygap) + 1, 5);
else
    acclog = zeros(floor(maxepoch / displaygap) + 1, 3);
end

fprintf(['Training starts...\n' ...
    'numhids: %d\n' ...
    'learnrate: %f\n' ...
    'nCD: %d\n' ...
    'momentum: %f\n' ...
    'weightcost: %f\n' ...
    'maxepoch: %d\n' ...
    'restartgap: %d\n' ...
    ], numhids, learnrate, nCD, momentum, weightcost, maxepoch, restartgap);

tic;

lambdav = lambda;
if lambdav == 0
    lambdav = 1;
end

% lbs = pixlabel_inf(unary, vishid, visbiases, hidbiases, 50, lambdav);
if initu
    lbs = pixlabel_inf_em_p(unary, pairwise, vishid, visbiases, hidbiases, 50, lambdav, mu, 1);
else
    lbs = pixlabel_inf_em_p(unary, pairwise, vishid, visbiases, hidbiases, 50, lambdav, mu, 0);
end
acc2 = mean(sum(lbs & data, 2) ./ sum(lbs | data, 2));
acc1 = sum(sum(lbs == data, 2)) / prod(size(data));
fprintf(['epoch %d, acc1 %.4f, acc2 %.4f, err %.2f, '], 0, acc1, acc2, 0);

acclog(1, 1) = 0;
acclog(1, 2) = acc1;
acclog(1, 3) = acc2;

if isval
    % lbs = pixlabel_inf(uval, vishid, visbiases, hidbiases, 50, lambdav);
    if initu
        lbs = pixlabel_inf_em_p(uval, pval, vishid, visbiases, hidbiases, 50, lambdav, mu, 1);
    else
        lbs = pixlabel_inf_em_p(uval, pval, vishid, visbiases, hidbiases, 50, lambdav, mu, 0);
    end
    acc2 = mean(sum(lbs & xval, 2) ./ sum(lbs | xval, 2));
    acc1 = sum(sum(lbs == xval, 2)) / prod(size(xval));
    fprintf(['tacc1 %.4f, tacc2 %.4f, '], acc1, acc2);
    acclog(1, 4) = acc1;
    acclog(1, 5) = acc2;
end

fprintf(['vhinc %.4f, vbinc %.4f, hbinc %.4f, time %.2f\n'], ...
    max(max(abs(vishidinc))), max(abs(visbiasinc)), max(abs(hidbiasinc)), toc);

for epoch = 1 : maxepoch
    errsum1 = 0;
    errsum2 = 0;
    errsum = 0;
    sampleloss = 0;

    % randomly reorder all the data cases
    idx = randperm(numtotalcases);
    % idx = 1:numtotalcases;

    if decgap > 0 && mod(epoch, decgap) == 0
        epsilonw  = learnrate / (epoch / decgap + 1);
        epsilonvb = learnrate / (epoch / decgap + 1);
        epsilonhb = learnrate / (epoch / decgap + 1);
    end

    if restartgap > 0 && mod(epoch, restartgap) == 0
        for isample = 1 : numsamples
            if initu
                visk{isample} = double(unarymx < 0);
            else
                visk{isample} = pixlabel_inf_crf(unary, pairwise, lbcost, mu);
            end
            hidprobs = 1 ./ (1 + exp(-visk{isample} * vishid - ones(numtotalcases, 1) * hidbiases));
            hidk{isample} = hidprobs > rand(numtotalcases, numhids);
        end
        vishidinc  = vishidinc * 0;
        hidbiasinc = hidbiasinc * 0;
        visbiasinc = visbiasinc * 0;

        restartgap = restartgap * 2;
    end

    for batch = 1 : numbatches
        istart = (batch - 1) * batchsz + 1;
        if batch ~= numbatches
            iend = istart + batchsz - 1;
        else
            iend = numtotalcases;
        end

        x = data(idx(istart:iend), :);
        unarybatch = unarymx(idx(istart:iend), :);
        pbatch = pairwise(idx(istart:iend), :);

        [numcases, numdims] = size(x);
        
        % sample h ~ h|x
        poshidprobs = 1 ./ (1 + exp(-x * vishid - ...
            ones(numcases, 1) * hidbiases));
        hid0 = poshidprobs > rand(numcases, numhids);

        %{
        recvisprobs = 1 ./ (1 + exp(unarybatch - hid0 * vishid' - ...
            ones(numcases, 1) * visbiases));
        recvis = recvisprobs > rand(numcases, numdims);
        %}

        [recvis, ~] = mrf_gibbs_mx(-unarybatch, pbatch, lbcost*lambda*mu, ...
            [32,32], x, ones(numcases,1)*visbiases + hid0 * vishid');

        loss = zeros(numcases, numsamples);

        for isample = 1 : numsamples
            visktemp = visk{isample}(idx(istart:iend), :);
            hidktemp = hidk{isample}(idx(istart:iend), :);

            for cd = 1 : nCD
                %{
                negvisprobs = 1 ./ (1 + exp(-hidktemp * vishid' - ...
                    ones(numcases, 1) * visbiases + unarybatch));
                visktemp = negvisprobs > rand(numcases, numdims);
                %}
                [visktemp, ~] = mrf_gibbs_mx(-unarybatch, pbatch, lbcost*lambda*mu, ...
                    [32,32], visktemp, ones(numcases,1)*visbiases + hidktemp * vishid');
                
                neghidprobs = 1 ./ (1 + exp(-visktemp * vishid - ...
                    ones(numcases, 1) * hidbiases));
                hidktemp = neghidprobs > rand(numcases, numhids);
            end

            viskbatch{isample} = visktemp;
            hidkbatch{isample} = hidktemp;
            hidprobsbatch{isample} = neghidprobs;

            visk{isample}(idx(istart:iend), :) = visktemp;
            hidk{isample}(idx(istart:iend), :) = hidktemp;

            if isiou % Intersection over union loss
                loss(:, isample) = 1 - sum(visktemp & x, 2) ./ sum(visktemp | x, 2);
            else % Per pixel loss
                loss(:, isample) = 1 - sum(visktemp == x, 2) / size(visktemp, 2);
            end
        end

        sampleloss = sampleloss + sum(mean(loss, 2));

        lossdiff = loss - mean(loss, 2) * ones(1, numsamples);
        prods = zeros(size(vishid));
        visactloss = zeros(size(visbiases));
        hidactloss = zeros(size(hidbiases));        % the hidact used for loss
        hidact = zeros(size(hidbiases));            % the real hidact
        visact = zeros(size(visbiases));

        for isample = 1 : numsamples
            lossv = lossdiff(:, isample) * ones(1, numdims);
            lossh = lossdiff(:, isample) * ones(1, numhids);

            prodlossx = viskbatch{isample} .* lossv;
            prodlossh = hidkbatch{isample} .* lossh;

            prods = prods + prodlossx' * hidkbatch{isample};
            visactloss = visactloss + sum(prodlossx);
            hidactloss = hidactloss + sum(prodlossh);

            hidact = hidact + sum(hidprobsbatch{isample});
            visact = visact + sum(viskbatch{isample});
        end

        % CD finished, calculate error signals

        vishidinct  = -epsilonw * prods / (numcases * numsamples);
        visbiasinct = -epsilonvb * visactloss / (numcases * numsamples);
        hidbiasinct = -epsilonhb * hidactloss / (numcases * numsamples);

        vishidinc  = momentum * vishidinc + vishidinct;
        visbiasinc = momentum * visbiasinc + visbiasinct;
        hidbiasinc = momentum * hidbiasinc + hidbiasinct;

       
        vishidinc = vishidinc - epsilonw * weightcost * vishid;

         % sparsity constraint
        hidact = hidact / (numcases * numsamples);
        visact = visact / (numcases * numsamples);
        if batch == 1
            hidactold = sparsetarget * ones(1, numhids);
        else
            hidactold = 0.9 * hidactold + 0.1 * hidact;
        end
        vishidinc = vishidinc - epsilonw * sparseweight * visact' * (hidactold - sparsetarget);
        hidbiasinc = hidbiasinc - epsilonhb * sparseweight * (hidactold - sparsetarget);
        
        vishid    = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
 
        vishid(vishid > 5) = 5;
        vishid(vishid < -5) = -5;
        visbiases(visbiases > 8) = 8;
        visbiases(visbiases < -8) = -8;

        err = sum(sum((x - recvis).^2));
        errsum = errsum + err;
    end

    if mod(epoch, displaygap) == 0
        lambdav = lambda;
        if lambdav == 0
            lambdav = 1;
        end

        % lbs = pixlabel_inf(unary, vishid, visbiases, hidbiases, 50, lambdav);
        if initu
            lbs = pixlabel_inf_em_p(unary, pairwise, vishid, visbiases, hidbiases, 50, lambdav, mu, 1);
        else
            lbs = pixlabel_inf_em_p(unary, pairwise, vishid, visbiases, hidbiases, 50, lambdav, mu, 0);
        end
        acc2 = mean(sum(lbs & data, 2) ./ sum(lbs | data, 2));
        acc1 = sum(sum(lbs == data, 2)) / prod(size(data));
        fprintf(['epoch %d, acc1 %.4f, acc2 %.4f, err %.0f, loss %.4f, '], epoch, acc1, acc2, errsum, sampleloss / numtotalcases);

        acclog(epoch/displaygap + 1, 1) = epoch;
        acclog(epoch/displaygap + 1, 2) = acc1;
        acclog(epoch/displaygap + 1, 3) = acc2;

        if isval
            % lbs = pixlabel_inf(uval, vishid, visbiases, hidbiases, 50, lambdav);
            if initu
                lbs = pixlabel_inf_em_p(uval, pval, vishid, visbiases, hidbiases, 50, lambdav, mu, 1);
            else
                lbs = pixlabel_inf_em_p(uval, pval, vishid, visbiases, hidbiases, 50, lambdav, mu, 0);
            end
            acc2 = mean(sum(lbs & xval, 2) ./ sum(lbs | xval, 2));
            acc1 = sum(sum(lbs == xval, 2)) / prod(size(xval));
            fprintf(['tacc1 %.4f, tacc2 %.4f, '], acc1, acc2);
            acclog(epoch/displaygap + 1, 4) = acc1;
            acclog(epoch/displaygap + 1, 5) = acc2;
        end

        fprintf(['vhinc %.4f, vbinc %.4f, hbinc %.4f, hact %.4f, time %.2f\n'], ...
            max(max(abs(vishidinc))), max(abs(visbiasinc)), max(abs(hidbiasinc)), mean(hidactold), toc);

        tic;
    end
    if isoutput && mod(epoch, savegap) == 0 
        save([outdir '/m' ...
            num2str(epoch) '.mat'], 'vishid', 'hidbiases', 'visbiases');
    end
    
    errlog(epoch, 1) = errsum;
end

if isoutput
    save([outdir '/model_err.mat'], 'errlog', 'acclog');
end

return
end

