function p = smooth_pairwise_uniform(imsz, imnum, verbose)
% SMOOTH_PAIRWISE
%
% Generate constant smoothing pairwise terms for a set of images.
%
% Usage:
% p = smooth_pairwise(imsz, imnum, verbose)
%
% Yujia Li, 10/2012
%

p = cell(imnum, 1);

pb = zeros(prod(imsz),1);

psingle = pairwise2pb(pb, imsz);
psingle = psingle / max(max(psingle));

for i = 1 : imnum
    p{i} = psingle;
end

return
end

