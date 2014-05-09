function pb = pairwise2pb(p, imsz)
% PAIRWISE2PB
%
% Usage:
% pb = pairwise2pb(p, imsz)
%
% pb is similar to probability of boundary, and is of the same size as imsz,
% this can be used to visualize pairwise potentials.
% 
% Yujia Li, 10/2012
%

if iscell(p)
    numcases = length(p);
    pb = zeros(numcases, prod(imsz));
    
    for i = 1 : numcases
        pb(i,:) = reshape(p2pb_single_fast(p{i}, imsz), [1,prod(imsz)]);

        if mod(i, 50) == 0
            fprintf('Processed %d images...\n', i);
        end
    end
else
    pb = p2pb_single_fast(p, imsz);
end

return
end

function pb = p2pb_single(p, imsz)
pb = zeros(imsz);
M = imsz(1);
N = imsz(2);

for i = 1 : M
    for j = 1 : N
        pidx = i + (j-1)*M;
        minpenalty = inf;

        if i > 1
            minpenalty = min(minpenalty, p(pidx, i-1+(j-1)*M));
        end
        if j > 1
            minpenalty = min(minpenalty, p(pidx, i+(j-2)*M));
        end
        if i < M
            minpenalty = min(minpenalty, p(pidx, i+1+(j-1)*M));
        end
        if j < N
            minpenalty = min(minpenalty, p(pidx, i+j*M));
        end

        pb(i,j) = minpenalty;
    end
end

pb = 1 ./ (1 + exp(pb));

return
end


function pb = p2pb_single_fast(p, imsz)

npix = size(p, 1);

pv = inf(npix, 1);

[ix, iy, v] = find(p);

for i = 1 : length(ix)
    idx = ix(i);
    pv(idx) = min(pv(idx), v(i));
end

pb = -reshape(pv, imsz);

return
end

