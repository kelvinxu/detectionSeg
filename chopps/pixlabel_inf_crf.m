function [lbs, obj] = pixlabel_inf_crf(unary, pairwise, lbcost, mu, extraunary)
% PIXLABEL_INF_CRF
%
% Infer pixel labels for every pixel in the image. Additional unary
% potentials can be plugged in here. A graph-cut optimizer is run here to
% compute the optimal labeling. Both single image inference and a batch of
% images are supported here.
%
% Usage:
% [lbs, obj] = pixlabel_inf_crf(unary, pairwise, lbcost, mu[, extraunary])
%
% lbs is a 1*NumSites vector.
% 
% Yujia Li, 04/2012
%

if iscell(unary)
    numimgs = length(unary);
    numsites = size(unary{1}, 2);

    lbs = cell(numimgs,1);
    for i = 1 : numimgs
        if nargin == 4
            lbs{i} = pixlabel_inf_crf_single(-unary{i}, pairwise{i}, lbcost * mu);
        else
            lbs{i} = pixlabel_inf_crf_single(-unary{i}, pairwise{i}, lbcost * mu, -extraunary{i});
        end
    end
    obj = 0;
    lbs = double(lbs == 2);
else
    if nargin == 4
        [y, obj] = pixlabel_inf_crf_single(-unary, pairwise, lbcost * mu);
    else
        [y, obj] = pixlabel_inf_crf_single(-unary, pairwise, lbcost * mu, -extraunary);
    end
    lbs{1} = double(y==2)';
end

return
end

function [lbs, obj] = pixlabel_inf_crf_single(unary, pairwise, lbcost, extraunary)

[numlabels, numsites] = size(unary);

h = GCO_Create(numsites, numlabels);
GCO_SetNeighbors(h, pairwise);
if nargin > 3
    GCO_SetDataCost(h, unary + extraunary);
else
    GCO_SetDataCost(h, unary);
end
GCO_SetSmoothCost(h, lbcost);
GCO_Expansion(h, 100);
lbs = GCO_GetLabeling(h);
if nargout >= 2
    obj = GCO_ComputeEnergy(h);
end
GCO_Delete(h);

return
end

