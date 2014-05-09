function p = combine_pairwise(plist, mulist)
% COMBINE_PAIRWISE
%
% Usage:
% p = combine_pairwise(plist, mulist)
%
% plist is a cell array, each element is a pairwise potential list.
% mulist is a 1-D vector.
%
% Yujia Li, 10/2012
%

numcases = length(plist{1});
numpairwise = length(plist);

p = cell(numcases, 1);

for i = 1 : numcases
    pp = plist{1}{i} * mulist(1);
    for j = 2 : numpairwise
        pp = pp + plist{j}{i} * mulist(j);
    end
    p{i} = pp;
end

return
end

