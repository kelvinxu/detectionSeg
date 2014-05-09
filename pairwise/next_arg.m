function [arg, argidx] = next_arg(arglist, current_argidx)
% NEXT_ARG
%
% Usage:
% [arg, argidx] = next_arg(arglist, current_argidx)
%
% Yujia Li, 10/2012
%

nargs = length(arglist);
idxlim = zeros(1, nargs);

arg = zeros(1, nargs);

if isempty(current_argidx)
    argidx = ones(1, nargs);
    for i = 1 : nargs
        arg(i) = arglist{i}(1);
    end
    return;
end

argidx = current_argidx;

for i = 1 : nargs
    idxlim(i) = length(arglist{i});
    arg(i) = arglist{i}(argidx(i));
end

i = 1;
while i <= nargs
    argidx(i) = argidx(i) + 1;
    if argidx(i) <= idxlim(i)
        arg(i) = arglist{i}(argidx(i));
        break;
    else
        argidx(i) = 1;
        arg(i) = arglist{i}(1);
        i = i + 1;
    end
end

if i > nargs
    arg = [];
end


return
end

