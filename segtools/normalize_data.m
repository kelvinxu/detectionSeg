function [normalized, tmean ,tstd] = normalize_data(data, tmean, tstd)
% helper function to standardize data
%
if ~iscell(data)
    normalized = cell(1);
    if nargin < 2
        [normalized{1}, tmean, tstd] = normalize_data_single(data);
    else
        [normalized{1}, ~,~] = normalize_data_single(data, tmean, tstd);
    end
else
    imnum = length(data);
    normalized = cell(1,imnum);
    for i = 1 : imnum
        if nargin < 2
            [normalized{i}, tmean, tstd] = normalize_data_single(data{i});
        else
            [normalized{i}, ~, ~] = normalize_data_single(data{i}, tmean, tstd);
        end
    end
end
end

function [normalized_single, tmean, tstd] = normalize_data_single(data, tmean, tstd)
    if nargin < 2
        tmean = mean(data);
        tstd = std(data);
    end 
    normalized_single = bsxfun(@minus,data,tmean);
    normalized_single = bsxfun(@rdivide, normalized_single, tstd);
end
