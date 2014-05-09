function mask = mrf_mask_gen(imsz)
% MRF_MASK_GEN
%
% Generate masks used in block Gibbs sampling for a 4-connected grid structure.
%
% Usage:
% mask = mrf_mask_gen(imsz)
%
% Yujia Li, 03/2012
%

mask = cell(2, 1);

im = zeros(imsz);

nx = imsz(1);
ny = imsz(2);

jstart = 1;
for i = 1 : nx
    for j = jstart : 2 : ny
        im(i, j) = 1;
    end
    jstart = 3 - jstart;
end

mask{1} = reshape(logical(im), [1, prod(imsz)]);
mask{2} = reshape(logical(ones(imsz) - im), [1, prod(imsz)]);

return
end
