function [ corrupted ] = corrupt_bndboxes( img, bndboxes )
%corrupt_bndboxes(img, bndboxes) adds noise to bndingboxes
%   This function corrupts the boxes with guassian noise with variance
%   equal to the length of the box

corrupted = bndboxes;

for i = 1:length(img)
    [rows, cols, ~] = size(img{i});
    for j = 1:length(bndboxes{i})
        notvalid = true;
        box = bndboxes{i}.box{j};
        while notvalid
            ymin = round(box.ymin + randn(1)/4*(box.ymax-box.ymin)^(0.5));
            ymax = round(box.ymax + randn(1)/4*(box.ymax-box.ymin)^(0.5));
            xmin = round(box.xmin + randn(1)/4*(box.xmax-box.xmin)^(0.5));
            xmax = round(box.xmax + randn(1)/4*(box.xmax-box.xmin)^(0.5));
            if (ymin >= 1 && ymax <= rows && xmin >= 1 && xmin <= cols)
                notvalid = false;
                corrupted{i}.box{j}.ymin = ymin; 
                corrupted{i}.box{j}.ymax = ymax;
                corrupted{i}.box{j}.xmin = xmin;
                corrupted{i}.box{j}.xmax = xmax;
            end 
        end 
    end
end
