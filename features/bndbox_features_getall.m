function [ bndboxfeatures ] = bndbox_features_getall(rows, cols, bxs)
% bndbox_features
% extracts some bounding box features from the dataset
% 
% The two features we have are
% 1) Whether or not the pixel is in the bounding box
% 2) The normalized distance in the bounding box

% In the future we are going to add features
nfeats = 6;

 bndboxfeatures = zeros([rows, cols, nfeats]);

% Sigma scaler which scales the guassian to the 
% side of the box
k_sigscaler = 0.25;

% 1)
isbox = zeros([rows, cols]);
if ~isempty(bxs)
for i=1:length(bxs.box)
    box = bxs.box{i};
    % get two boundinb boxes, one with noise and one without. 
        notvalid = true;
        while notvalid
            ymin = round(box.ymin + randn(1)*((box.ymax-box.ymin))^(0.5)/5);
            ymax = round(box.ymax + randn(1)*((box.ymax-box.ymin))^(0.5)/5);
            xmin = round(box.xmin + randn(1)*((box.ymax-box.ymin))^(0.5)/5);
            xmax = round(box.xmax + randn(1)*((box.ymax-box.ymin))^(0.5)/5);
        if (ymin >= 1 && ymax <= rows && xmin >= 1 && xmax <= cols && (xmin ~= xmax) && (ymin ~= ymax))
            notvalid = false;
        end 
        end 
        nnymin = box.ymin;
        nnymax = box.ymax;
        nnxmin = box.xmin;
        nnxmax = box.xmax;

% Smooth mask
    yrange = (nnymax-nnymin+1);
    xrange = (nnxmax-nnxmin+1);
    xgauss = repmat(fspecial('gaussian', [1,xrange], xrange*k_sigscaler), yrange,1);
    ygauss = repmat(fspecial('gaussian', [yrange,1], yrange*k_sigscaler), 1, xrange);
    bndboxfeatures(nnymin:nnymax, nnxmin:nnxmax,1) = bndboxfeatures(nnymin:nnymax, nnxmin:nnxmax,1) + ygauss .* xgauss;
 
%Smooth mask noise
    yrange = (ymax-ymin+1);
    xrange = (xmax-xmin+1);
    xgauss = repmat(fspecial('gaussian', [1,xrange], xrange*k_sigscaler), yrange,1);
    ygauss = repmat(fspecial('gaussian', [yrange,1], yrange*k_sigscaler), 1, xrange);
    bndboxfeatures(ymin:ymax, xmin:xmax,2) = bndboxfeatures(ymin:ymax, xmin:xmax,2) + ygauss .* xgauss;

%Hard Mask
    bndboxfeatures(nnymin:nnymax, nnxmin:nnxmax,3) = bndboxfeatures(nnymin:nnymax, nnxmin:nnxmax,3) + 1;

% Hard Mask Noise
    bndboxfeatures(ymin:ymax, xmin:xmax,4) = bndboxfeatures(ymin:ymax, xmin:xmax,4) + 1;
%Mask 
% using prior mask
    mask = load('mask_aeroplane.mat');
    bndboxfeatures(nnymin:nnymax, nnxmin:nnxmax,5) = bndboxfeatures(nnymin:nnymax, nnxmin:nnxmax,5) + imresize(mask.normalized, [nnymax-nnymin+1, nnxmax-nnxmin+1]);
    
% Mask with noise
    bndboxfeatures(ymin:ymax, xmin:xmax,6) = bndboxfeatures(ymin:ymax, xmin:xmax,6) + imresize(mask.normalized, [ymax-ymin+1, xmax-xmin+1]);

end 
end
end

