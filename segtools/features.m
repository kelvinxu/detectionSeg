function features = features(testim, testbndboxes)
% Gets cell of features based upon whether bndboxes are present or not
% Usage: features = features(testim, testbndboxes)
% testim/testbndboxes should be the same length, and should both be cell arrays

if nargin >1
    features = filter_response(testim, testbndboxes, 1);
else
    features = filter_response(testim);
end
end

