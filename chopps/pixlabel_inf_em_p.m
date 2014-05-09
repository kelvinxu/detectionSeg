function [lbs, sc, hprobs, visprobs] = pixlabel_inf_em_p(unary, p, vh, vb, hb, nIter, lambda, mu, initu)
% PIXLABEL_INF_EM_P
%
% Perform alternating inference to get all pixel labels.
%
% Usage: 
% [lbs, sc, hprobs, visprobs] = pixlabel_inf_em_p(unary, p, vh, vb, hb, nIter, lambda, mu[, initu])
%
% Yujia Li, 07/2012
%

if nargin >= 9 && initu
    lbs = inf_unary(unary);
else
    lbs = pixlabel_inf_crf(unary, p, [0,1;1,0], mu);
end
visprobs = zeros(size(lbs));
v = double(lbs == 1);
% v = init;
% v = zeros(size(lbs));
% v = ones(size(lbs));

if iscell(unary) % nargout <= 1
    % batch mode pixel label inference
    numcases = length(unary);

    for i = 1 : numcases
        unary{i} = unary{i} * lambda;
    end

    updatemask = logical(ones(numcases, 1));

    for i = 1 : nIter
        if i > 1
            if ~any(any(v ~= vnew))
                break;
            else
                updatemask = any(v ~= vnew, 2);
                v = vnew;
            end
        end
        hprobs = 1 ./ (1 + exp(-v(updatemask,:) * vh - ones(sum(updatemask),1)*hb)); 
        exunary = get_extraunary(vb, vh, hprobs);
        if ~iscell(exunary)
            exunary = {exunary};
        end

        % lbs(updatemask,:) = inf_unary(unary(updatemask), exunary);
        lbs(updatemask,:) = pixlabel_inf_crf(unary(updatemask), p(updatemask), [0,1;1,0], mu * lambda, exunary);
        vnew = double(lbs == 1);
        
        if nargout >= 4
            n = 0;
            for j = 1 : numcases
                if updatemask(j)
                    n = n + 1;
                    visprobs(j,:) = 1 ./ (1 + exp(exunary{n}(1,:)));
                end
            end
        end
    end
    hprobs = 1 ./ (1 + exp(-v * vh - ones(numcases,1)*hb)); 
else
    sc = [];
    [ir, jc, vp] = find(p);
    
    % hidrec = [];
    for i = 1 : nIter
        if i > 1
            if ~any(v ~= vnew)
                break;
            else
                v = vnew;
            end
        end
        hprobs = 1 ./ (1 + exp(-v * vh - hb)); 
        % hidrec = [hidrec; hprobs];

        % this is the primal form, but it is actually the same as the bound,
        % because now the q distribution is exactly the posterior distribution
        bound = v * (vb + lambda * (unary(2,:) - unary(1,:)))' + sum(log(1 + exp(hb + v * vh))) - mu * [v(ir) ~= v(jc)] * vp / 2;

        sc = [sc; bound];
        fprintf('%f\n', bound);

        exunary = get_extraunary(vb, vh, hprobs);

        % lbs = inf_unary(unary * lambda, exunary);
        lbs = pixlabel_inf_crf(unary * lambda, p, [0,1;1,0], mu * lambda, exunary);
        visprobs = 1 ./ (1 + exp(exunary(1,:)));
        vnew = double(lbs == 1);
    end
end

return
end

function lbs = inf_unary(unary, exunary)

if iscell(unary)
    numcases = length(unary);
    numsites = size(unary{1}, 2);

    lbs = zeros(numcases, numsites);
    for i = 1 : numcases
        if nargin > 1
            lbs(i,:) = (unary{i}(1,:) + exunary{i}(1,:)) <= (unary{i}(2,:) + exunary{i}(2,:));
        else
            lbs(i,:) = (unary{i}(1,:) <= unary{i}(2,:));
        end
    end
else
    if nargin > 1
        lbs = (unary(1,:) + exunary(1,:)) <= (unary(2,:) + exunary(2,:));
    else
        lbs = (unary(1,:) <= unary(2,:));
    end
end

return
end

