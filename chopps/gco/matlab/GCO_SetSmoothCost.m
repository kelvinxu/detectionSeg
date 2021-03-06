function GCO_SetSmoothCost(Handle,SmoothCost)
% GCO_SetSmoothCost   Set the smooth cost of neighboring sites.
%    GCO_SetSmoothCost(Handle,SmoothCost) with a NumLabels-by-NumLabels 
%    integer matrix makes SmoothCost(k,l) the unweighted cost 
%    of assigning labels k and l to any neighboring sites. 
%    For particular neighboring sites i,j the final, weighted cost
%    is actually Weights(i,j)*SmoothCost(k,l). The spatially-varying 
%    weights are determined by GCO_SetNeighbors.
%
%    If SetSmoothCost is never called, Potts model is used by default.
%    i.e. SmoothCost(k,l) = { 0 if k==l, 1 otherwise }
%
%    SetSmoothCost can be called repeatedly, even after Expansion.

GCO_LoadLib();
if (nargin < 2)
    error('Expected 2 arguments');
end
if (~isnumeric(SmoothCost))
    error('SmoothCost must be numeric');
end
NumLabels = gco_matlab('gco_getnumlabels',Handle);
if (size(SmoothCost) ~= [ NumLabels NumLabels ])
    error('SmoothCost size must be [ NumLabels NumLabels ]');
end
if ~isa(SmoothCost,'int32')
    if (NumLabels > 50 || any(any(floor(SmoothCost) ~= SmoothCost)))
        % warning('GCO:int32','SmoothCost converted to int32');
    end
    % SmoothCost = int32(SmoothCost);
end
% Make sure smooth cost is double
SmoothCost = double(SmoothCost);
if (any(SmoothCost ~= SmoothCost'))
    error('SmoothCost must be symmetric');
end
gco_matlab('gco_setsmoothcost',Handle,SmoothCost);
end
