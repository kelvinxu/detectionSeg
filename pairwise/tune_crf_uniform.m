function acclog = tune_crf(unary, pairwise, imgs, argsrange, groundtruth, txtout)
% TUNE_CRF(for binary segmentation)
%
% Usage:
% acclog = tune_crf(unary, pairwise, imgs, argsrange, groundtruth[, txtout])
%
% Test all argument settings in the given range and print out pix and iou accuracy.
%
% unary is a cell array, the usual unary potentials used.
%
% pairwise is a cell array, each element is a pairwise potential list as usually
%       used. But the length of this array is actually the number of different
%       types of pairwise potentials minus 1 (npairwise-1), because the Boykov 
%       & Jolly potential will be generated on the fly.
% imgs is a list of images, cell array too.
% argsrange is cell array of length npairwise+1, the first npairwise elements are
%       ranges for mu parameters for corresponding pairwise type. The last is the 
%       range for sigma used in Boykov & Jolly pairwise term.
%       
%       each entry of argrange is a matlab range like 1:4:10, (so an actual array)
%
% groundtruth is a matrix for ground truth output. Each row is a data case.
% using the matlab reshape function 
%
%
% Yujia Li, 10/2012
%

% we add in the boykov jolly potentials
npairwise = length(pairwise) + 1;
% narg is a list of the lengths of the mu parameters
nargs = zeros(1, npairwise);

for i = 1 : npairwise
    nargs(i) = length(argsrange{i});
end

%plist just loads in the pairwise lists
plist = cell(1, npairwise);
for i = 1 : npairwise - 1
    plist{i} = pairwise{i};
end

% range of sigmas for Boykov & Jolly pairwise terms
nsigma = length(argsrange{npairwise + 1});

% ntest holds the number of different combinations of narg
% nargs is a column 
ntest = prod(nargs);

% add three since the top three rows are reserved for accuracy
% results 
acclog = zeros(npairwise + 3, ntest * nsigma);

maxacc = 0;
optargidx = 0;

% if the input array dimensions suggest that we don't have any
% boykov jolly params 
if length(argsrange{npairwise}) == 1 && argsrange{npairwise}(1) == 0
    useboykov = 0;
else
    useboykov = 1;
end

for j = 1 : nsigma

    tic;
    sigma = argsrange{npairwise + 1}(j);
    if useboykov
        fprintf('Generating boykov & jolly pairwise, for sigma = %.2f...', sigma);
        pboykov = boykov_pairwise(imgs, sigma, 0);
        fprintf('Done... time %.2f\n', toc);
        plist{npairwise} = pboykov;
    end

    argidx = [];
    tic;
    for i = 1 : ntest
        [mu, argidx] = next_arg(argsrange(1:npairwise), argidx);
        if useboykov 
            p = combine_pairwise(plist, mu);
        else
            p = combine_pairwise(plist(1:npairwise-1), mu(1:npairwise-1));
        end
        
        % if unary is a cell(as is often the case here)
        % pixlabel_inf_crf will return a matrix where every row is a datacase(read:image labeling)      
        y = pixlabel_inf_crf(unary, p, [0,1;1,0], 1);
        pixacc = sum(sum(y == groundtruth)) / prod(size(y));
        iouacc = mean(sum(y & groundtruth, 2) ./ sum(y | groundtruth, 2));
        acclog(1,i+(j-1)*ntest) = pixacc;
        acclog(2,i+(j-1)*ntest) = iouacc;
        acclog(3:npairwise+2, i+(j-1)*ntest) = mu;
        acclog(npairwise+3,i+(j-1)*ntest) = sigma;

        if pixacc + iouacc > maxacc
            maxacc = pixacc + iouacc;
            optargidx = i+(j-1)*ntest;
        end

        for k = 1 : npairwise
            fprintf('mu%d= %.2f, ', k, mu(k));
        end
        fprintf(' sigma= %.2f, pixacc %.4f, iouacc %.4f, etime %.4f\n', sigma, pixacc, iouacc, toc);
        tic;
    end
end

fprintf('------------------------------------------------------\nBest parameter setting found:\n');
for k = 1 : npairwise
    fprintf('mu%d= %.2f, ', k, acclog(2+k, optargidx));
end
fprintf(' sigma= %.2f, pixacc %.4f, iouacc %.4f, etime %.4f\n', ...
    acclog(npairwise+3, optargidx), acclog(1,optargidx), acclog(2,optargidx), toc);

if nargin >= 6
    f = fopen(txtout, 'w');
    fprintf(f, '------------------------------------------------------\nBest parameter setting found:\n');
    for k = 1 : npairwise
        fprintf(f, 'mu%d= %.2f, ', k, acclog(2+k, optargidx));
    end
    fprintf(f, ' sigma= %.2f, pixacc %.4f, iouacc %.4f, etime %.4f\n', ...
        acclog(npairwise+3, optargidx), acclog(1,optargidx), acclog(2,optargidx), toc);
    fclose(f);
end

return
end

