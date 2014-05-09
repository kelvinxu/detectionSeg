function unary = get_extraunary(vb, vh, hprobs)
% GET_EXTRAUNARY
%
% Compute the extra unary potentials induced by the shape model.
%
% Usage:
% unary = get_extraunary(vb, vh1, h1probs)
%
% Yujia Li, 04/2012
%

numdims = length(vb);

[numcases, numh] = size(hprobs);

visact = - hprobs * vh' - ones(numcases, 1) * vb;
% visact = - hprobs * vh';

% if nargin > 3
%     visact = visact * lambda;
% end

if numcases == 1
    unary = zeros(2, numdims);
    % unary(2,:) are set to be 0
    unary(1,:) = visact;
else
    unary = cell(numcases, 1);
    for i = 1 : numcases
        unary{i} = zeros(2, numdims);
        unary{i}(1,:) = visact(i,:);
    end
end

return
end

