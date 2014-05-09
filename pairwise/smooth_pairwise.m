function p = smooth_pairwise(imgs, verbose)
% SMOOTH_PAIRWISE
%
% Generate constant smoothing pairwise terms for a set of images.
%
% Usage:
% p = smooth_pairwise(imgs, verbose)
%
% Kelvin Xu, 05/2014
%

imnum = length(imgs);
p = cell(imnum, 1);

for i = 1 : imnum
    [nx, ny, ~] = size(imgs{i});
    pb = zeros(prod([nx,ny]),1);
    
    psingle = pairwise2pb(pb, [nx,ny]);
    psingle = psingle / max(max(psingle));

    p{i} = psingle;
    if verbose
        sprintf('Done %d images\n', i);
    end
end

end

