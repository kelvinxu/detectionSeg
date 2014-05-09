function p = pb2pairwise(pb, verbose)
% PB2PAIRWISE
%
% Convert the boundary detection responses into pairwise weights used in CRF.
% 
% Usage:
% p = pb2pairwise(pb, verbose)
%
% pb can be for only one image, it can also be a cell array containing pb for
% a list of images.
%
% Yujia Li, 04/2012
%

if ~iscell(pb)
    p = pb2pairwise_single(pb);
else
    numcases = length(pb);
    p = cell(numcases, 1);
    for i = 1 : numcases
        p{i} = pb2pairwise_single(pb{i});

        if verbose && mod(i, 100) == 0
            fprintf('Converted %d PBs...\n', i);
        end
    end
end

return
end


function p = pb2pairwise_single(pb)

[nx, ny] = size(pb);
npix = nx * ny;

epsilon = 1e-10;

% Larger PB means less penalty
% v = log((1 - pb + epsilon) ./ (pb + epsilon));
v = -log(pb + epsilon);
% v = 1 - pb;

p = sparse(npix, npix);
for j = 1 : ny
    for i = 1 : nx
        pidx = i + (j - 1) * nx;

        if j < ny
            pnbidx = i + j * nx;
            p(pidx, pnbidx) = min(v(i,j), v(i,j+1));
        end
        if i > 1
            pnbidx = i - 1 + (j - 1) * nx;
            p(pidx, pnbidx) = min(v(i,j), v(i-1,j));
        end
        if j > 1
            pnbidx = i + (j - 2) * nx;
            p(pidx, pnbidx) = min(v(i,j), v(i,j-1));
        end
        if i < nx
            pnbidx = i + 1 + (j - 1) * nx;
            p(pidx, pnbidx) = min(v(i,j), v(i+1,j));
        end
    end
end

return
end

