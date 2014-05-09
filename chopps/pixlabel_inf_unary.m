function lbs = pixlabel_inf_unary(unary)
% PIXLABLE_INF_UNARY
%
% lbs = pixlabel_inf_unary(unary)
%
% Yujia Li, 08/2012
%

if iscell(unary)
    numcases = length(unary);
    numdims = size(unary{1}, 2);

    lbs = zeros(numcases, numdims);

    for i = 1 : numcases
        lbs(i,:) = unary{i}(2,:) > unary{i}(1,:);
    end
else
    lbs = unary(2,:) > unary(1,:);
end

return
end
