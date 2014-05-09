function scaled_box = scale_box(box, imsz, s_imsz)
%
% scale_box(box, imsz)
%
% This function takes a bounding box, an image size, and a 
% scaled image size
%

xscale = imsz(2)/s_imsz(2);
yscale = imsz(1)/s_imsz(1);

scaled_box.xmin = max(round(box.xmin / xscale), 1);
scaled_box.xmax = min(round(box.xmax / xscale), imsz(2)); 
scaled_box.ymin = max(round(box.ymin / yscale), 1); 
scaled_box.ymax = min(round(box.ymax / yscale), imsize(1));
end
