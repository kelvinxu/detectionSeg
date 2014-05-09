function [ bndboxfeatures ] = bndbox_features( rows, cols, bxs, isnoise, featnum)
% bndbox_features
% extracts some bounding box features from the dataset
% 
% The two features we have are
% 1) Whether or not the pixel is in the bounding box
% 2) The normalized distance in the bounding box

%
% featnum
% 1: smooth
% 2: smooth noise. 
% 3: hard
% 4: hard noise
% 5: mask
% 6: mask noise


nfeats = 1;

 bndboxfeatures = zeros([rows*cols, nfeats]);

% Sigma scaler which scales the guassian to the 
% side of the box
k_sigscaler = 0.25;

% 1)
isbox = zeros([rows, cols]);
for i=1:length(bxs.box)
    box = bxs.box{i};
    %noise term add in a guassian noise term with variance the length of
    % ground truth bounding box 
    if isnoise
        notvalid = true;
        while notvalid
        ymin = round(box.ymin + randn(1)*(box.ymax-box.ymin)^(0.5)/5);
        ymax = round(box.ymax + randn(1)*(box.ymax-box.ymin)^(0.5)/5);
        xmin = round(box.xmin + randn(1)*(box.xmax-box.xmin)^(0.5)/5);
        xmax = round(box.xmax + randn(1)*(box.xmax-box.xmin)^(0.5)/5);
        if (ymin >= 1 && ymax <= rows && xmin >= 1 && xmax <= cols)
            notvalid = false;
        end 
        end 
    else
        ymin = box.ymin;
        ymax = box.ymax;
        xmin = box.xmin;
        xmax = box.xmax;
    end
% Smooth mask
    yrange = (ymax-ymin+1);
    xrange = (xmax-xmin+1);
    xgauss = repmat(fspecial('gaussian', [1,xrange], xrange*k_sigscaler), yrange,1);
    ygauss = repmat(fspecial('gaussian', [yrange,1], yrange*k_sigscaler), 1, xrange);
    isbox(ymin:ymax, xmin:xmax) = ygauss .* xgauss; % 2-D guassian. 
    bndboxfeatures(1,
end 
% isbox (ymin:ymax, xmin:xmax) = fspecial('gaussian', [yrange, xrange], 10);
% create soft-mask on data
%         yrange = (ymin + ymax)/2;
%         xrange = (xmin + xmax)/2;
%         for i = xmin:xmax
%             for j = ymin:ymax
%                 some kind of smoother bounding box feature
%             end
%         end
        
%     hard-mask
        isbox(ymin:ymax,xmin:xmax) = isbox(ymin:ymax,xmin:xmax) ...
        + 1;
bndboxfeatures = isbox;

% using prior mask
% mask = load('mask_aeroplane.mat');
% isbox(ymin:ymax,xmin:xmax) = isbox(ymin:ymax,xmin:xmax) + imresize(mask.normalized, [ymax-ymin+1, xmax-xmin+1]);
% bndboxfeatures = isbox; 

end
end

