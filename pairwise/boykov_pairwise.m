function p = boykov_pairwise(imgs, sigma, verbose)
% BOYKOV_PAIRWISE
%
% Compute the pairwise potentials introduced in Boykov & Jolly ICCV01, where
% p(i,j) = exp(-(I_i - I_j)^2/(2*sigma^2))
% 
% Usage:
% p = boykov_pairwise(imgs, sigma[, verbose])
%
% imgs can be for only one image, it can also be a cell array containing 
% a list of images.
%
% Yujia Li, 10/2012
%

if nargin < 3
    verbose = 1;
end

if ~iscell(imgs)
    p = boykov_pairwise_single_fast(imgs, sigma);
else
    numcases = length(imgs);
    p = cell(numcases, 1);
    for i = 1 : numcases
        p{i} = boykov_pairwise_single_fast(imgs{i}, sigma);

        if mod(i, 100) == 0 && verbose
            fprintf('Generated %d pairwise potentials...\n', i);
        end
    end
end

return
end


function p = boykov_pairwise_single(img, sigma)

[nx, ny, ~] = size(img);
npix = nx * ny;

var = 2 * sigma * sigma;

v = double(img);

p = sparse(npix, npix);
for j = 1 : ny
    for i = 1 : nx
        pidx = i + (j - 1) * nx;

        if j < ny
            pnbidx = i + j * nx;
            p(pidx, pnbidx) = exp(-sum(((v(i,j,:) - v(i,j+1,:)) / 255).^2) / var);
        end
        if i > 1
            pnbidx = i - 1 + (j - 1) * nx;
            p(pidx, pnbidx) = exp(-sum(((v(i,j,:) - v(i-1,j,:)) / 255).^2) / var);
        end
        if j > 1
            pnbidx = i + (j - 2) * nx;
            p(pidx, pnbidx) = exp(-sum(((v(i,j,:) - v(i,j-1,:)) / 255).^2) / var);
        end
        if i < nx
            pnbidx = i + 1 + (j - 1) * nx;
            p(pidx, pnbidx) = exp(-sum(((v(i,j,:) - v(i+1,j,:)) / 255).^2) / var);
        end
    end
end

return
end


function p = boykov_pairwise_single_fast(img, sigma)

[nx, ny, ~] = size(img);

npix = nx * ny;
nedges = 2 * (nx * (ny - 1) + (nx - 1) * ny);

var = 2 * sigma * sigma;

v = reshape(double(img), [npix, size(img, 3)]);

idx_from = zeros(nedges, 1);
idx_to = zeros(nedges, 1);
pv = zeros(nedges, 1);

n_counted = 0;
horizontal_base = 1:nx:(npix-nx);
vertical_base = 1:(nx-1);

% horizontal edges to the left
for i = 1 : nx
    idx_from(n_counted + 1 : n_counted + (ny-1)) = horizontal_base + (i-1);
    idx_to(n_counted + 1: n_counted + (ny-1)) = horizontal_base + (i-1) + nx;
    n_counted = n_counted + ny - 1;
end

% vertical edges downward
for i = 1 : ny
    idx_from(n_counted + 1 : n_counted + (nx - 1)) = vertical_base + (i-1) * nx;
    idx_to(n_counted + 1 : n_counted + (nx - 1)) = vertical_base + (i-1) * nx + 1;
    n_counted = n_counted + nx - 1;
end

% add edges in the other directions.
idx_from(n_counted + 1 : nedges) = idx_to(1 : n_counted);
idx_to(n_counted + 1 : nedges) = idx_from(1 : n_counted);

pv = exp(-sum(((v(idx_from,:) - v(idx_to,:)) / 255).^2, 2) / var);

p = sparse(idx_from, idx_to, pv, npix, npix);

return
end

