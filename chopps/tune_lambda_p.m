function [acclog, bestlambda] = tune_lambda_p(unary, pairwise, vh, vb, hb, mu, lambdarange, groundtruth)
% TUNE_LAMBDA_P
%
% Usage:
% [acclog, bestlambda] = tune_lambda_p(unary, pairwise, vh, vb, hb, mu, lambdarange, groundtruth)
%
% Test all lambdas in the given range and print out pix and iou accuracy.
%
% Yujia Li, 10/2012
%

ntest = length(lambdarange);

acclog = zeros(3, ntest);

bestacc = 0;
bestiou = 0;
bestpix = 0;
bestlambda = 0;

tic;
for i = 1 : ntest
    lambda = lambdarange(i);
    
    y = pixlabel_inf_em_p(unary, pairwise, vh, vb, hb, 50, lambda, mu);
    pixacc = sum(sum(y == groundtruth)) / prod(size(y));
    iouacc = mean(sum(y & groundtruth, 2) ./ sum(y | groundtruth, 2));
    acclog(1,i) = mu;
    acclog(2,i) = pixacc;
    acclog(3,i) = iouacc;

    if pixacc + iouacc > bestacc
        bestpix = pixacc;
        bestiou = iouacc;
        bestacc = pixacc + iouacc;
        bestlambda = lambda;
    end

    fprintf('lambda= %.2f, pixacc %.4f, iouacc %.4f, etime %.4f\n', lambda, pixacc, iouacc, toc);
    tic;
end

fprintf('------------------------------------\n');
fprintf('Best lambda: %.2f, pixacc %.4f, iouacc %.4f\n', bestlambda, bestpix, bestiou);

return
end

