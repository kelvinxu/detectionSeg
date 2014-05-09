function p = smooth_pairwise(imsz, imnum, verbose)
% SMOOTH_PAIRWISE
%
% Generate constant smoothing pairwise terms for a set of uniform size images.
%
% Usage:
% p = smooth_pairwise(imsz, imnum, verbose)
%
% Yujia Li, 10/2012
%

p = cell(imnum, 1);

pb = zeros(imsz);

psingle = pb2pairwise(pb);
psingle = psingle / max(max(psingle));

for i = 1 : imnum
    p{i} = psingle;
end

return
end

