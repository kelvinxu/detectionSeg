function [ im,labeling, bndboxes ] = load_class(class_name, imgset, max_images, VOCopts)
%LOAD_CLASS: loads images from VOC
% 
% load_class(class_name, imgset, max_images, VOCopts)
% 
% class_name is the name of following the VOC convention 
%            'car', 'aeroplane', etc
%
% max_image is the maximum number of images to extract
% imageset is either 'train', 'trainval', or 'test'

im = {};
labeling = {};
bndboxes = {};
[ids,gt]=textread(sprintf(VOCopts.clsimgsetpath, ...
class_name,imgset),'%s %d');

[~,index_sorted] = sort(-gt);
ids = ids(index_sorted);
gt = gt(index_sorted);

pos_img = sum(gt(gt == 1));

% Go through all the images, and check to see if a segmentation file
% exist, if it does, then load the labeling and the image into the return
% cells
y = 1;
for i = 1:length(ids)
    if exist(sprintf(VOCopts.seg.clsimgpath, ids{i}))
        im{y} = imread(sprintf(VOCopts.imgpath, ids{i}));
        labeling{y} = importdata((sprintf(VOCopts.seg.clsimgpath, ids{i})));
        
        %read in annotation
        rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
        for z = 1:length(rec.objects)
            bndboxes{y}.box{z} = rec.objects(1,z).bndbox;
        end
        y = y+1;
    end
    if y > max_images
        break
    end
end
end

    