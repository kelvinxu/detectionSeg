function resp = applyfilterbank(im, cform, lmfilter, sfilter, extrafeats, boxes, isnoise)
% APPLYFILTERBANK
%
% Apply a bank of filters to an image and collect responses.
%
% Usage:
% resp = applyfilterbank(im, cform, lmfilter, sfilter)
%
% Yujia Li, 04/2012
%

dimrgb = 3;
dimlab = 3;
dimgabor = 40;
dimlm = 48;
dims = 13;

numdims = dimrgb + dimlab + dimgabor + dimlm + dims;

%extra features for later
if nargin > 4
   numdims = numdims + extrafeats; 
end

[nx, ny, nc] = size(im);

if nc == 1
    newim = zeros(nx, ny, 3);
    for i = 1 : 3
        newim(:,:,i) = im;
    end
    im = uint8(newim);
end

resp = zeros(nx, ny, numdims);

didx = 0;

% RGB colors
resp(:,:,didx+1:didx+dimrgb) = single(im) / 255;
didx = didx + dimrgb;

% LAB colors
if nargin < 2
    cform = makecform('srgb2lab');
end
labim = applycform(im, cform);
resp(:,:,didx+1:didx+dimlab) = single(labim) / 255;
didx = didx + dimlab;

% convert color image into gray image to compute Gabor features
dbim = sum(double(im) / 255, 3) / 3;

gbfilter = sg_createfilterbank(size(dbim), 0.2, 5, 4);
gbresp = sg_filterwithbank(dbim, gbfilter, 'method', 1);
gbresp = sg_resp2samplematrix(gbresp);
resp(:,:,didx+1:didx+dimgabor/2) = real(gbresp);
resp(:,:,didx+dimgabor/2+1:didx+dimgabor) = imag(gbresp);
didx = didx + dimgabor;

% LM filters
if nargin < 3
    lmfilter = makeLMfilters();
end
for i = 1 : dimlm
    resp(:,:,didx+i) = conv2(dbim, lmfilter(:,:,i), 'same');
end
didx = didx + dimlm;

% S filters
if nargin < 4
    sfilter = makeSfilters();
end
for i = 1 : dims
    resp(:,:,didx+i) = conv2(dbim, sfilter(:,:,i), 'same');
end
didx = didx + dims;

% BndBox Features
if nargin > 4
resp(:,:, didx+1:didx+extrafeats) = bndbox_features_getall(nx, ny, boxes);
end 

resp = reshape(resp, [nx * ny, numdims]);

return
end

