function patchviewx(plist, psize, nrow, isrowmajor, iscolor, whitesep, ...
    whitegrid, issc)
% PATCHVIEW
% 
% Display patch images stored in plist
%
% Usage: patchviewx(plist, psize, nrow, isrowmajor, iscolor, whitesep, ...
%    whitegrid)
%
% plist: N * D, N is number of images, D is the dimensionality
% psize: D=psize(1)*psize(2) or D=psize(1)*psize(2)*3 if it is color image
% nrow: number of rows of the patch view 
% isrowmajor: image organized in a row major order or not.
% iscolor: 1 if image is a color image
% whitesep: insert 3 white pixels between rows
% whitegrid: if 1, then use white grids rather than black ones.
% issc: if 1, use imagesc rather than imshow
%
% Yujia Li, 1/6/2012
%

if nargin < 4
    isrowmajor = 0;
end
if nargin < 5
    iscolor = 0;
end
if nargin < 6
    whitesep = 0;
end
if nargin < 7
    whitegrid = 0;
end
if nargin < 8
    issc = 0;
end

[np] = size(plist, 1);

ncol = ceil(np / nrow);

sizex = psize(1);
sizey = psize(2);


if whitesep
    if iscolor
        img = zeros(nrow * sizex + (nrow - 1) * 4 + 2, ncol * sizey + ncol + 1, 3);
    else
        img = zeros(nrow * sizex + (nrow - 1) * 4 + 2, ncol * sizey + ncol + 1);
    end
else
    if iscolor
        img = zeros(nrow * sizex + nrow + 1, ncol * sizey + ncol + 1, 3);
    else
        img = zeros(nrow * sizex + nrow + 1, ncol * sizey + ncol + 1);
    end
end
if whitegrid
    if iscolor
        img = img + 255;
    else
        img = img + max(max(double(plist)));
    end
end

ibase = 1;
in = 0;
for i = 1 : nrow
    jbase = 1;
    for j = 1 : ncol
        in = in + 1;
        if in > np
            break;
        end
        if iscolor
            if isrowmajor
                v = reshape(plist(in, :), [sizex, sizey, 3]);
                for k = 1 : 3
                    v(:, :, k) = v(:, :, k)';
                end
                img(ibase + 1: ibase + sizex, jbase + 1 : jbase + sizey, :) = v;
            else
                img(ibase + 1: ibase + sizex, jbase + 1 : jbase + sizey, :) ...
                    = reshape(plist(in, :), [sizex, sizey, 3]);
            end
        else
            if isrowmajor
                for px = 1 : sizex
                    for py = 1 : sizey
                        img(ibase + px, jbase + py) = ...
                            plist(in, (px - 1) * sizey + py);
                    end
                end
            else
                for py = 1 : sizey
                    for px = 1 : sizex
                        img(ibase + px, jbase + py) = ...
                            plist(in, (py - 1) * sizex + px);
                    end
                end
            end
        end
        jbase = jbase + sizey + 1;
    end
    ibase = ibase + sizex + 1;
    if whitesep && i < nrow
        if iscolor
            img(ibase + 1 : ibase + 2, :, :) = 1 * 3500;
        else
            img(ibase + 1 : ibase + 2, :) = 1 * 3500;
        end
        ibase = ibase + 3;
    end
end

if iscolor
    imshow(uint8(img)); %, [-5 5]);
else
    if issc
        imagesc(img); %, [-5 5]);
        axis off;
    else
        imshow(double(img), []); %, [-5 5]);
    end
end

return
end
