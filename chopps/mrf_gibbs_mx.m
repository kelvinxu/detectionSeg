function [x, xexpt] = mrf_gibbs_mx(unary, pairwise, lbcost, gridsz, initx, ...
    extraunary)
% MRF_GIBBS_MX
%
% One step block Gibbs sampling in a grid-structured MRF with unary and 
% pairwise terms. Matrix version of the mrf_gibbs function, which should be
% more efficient than the cell array version.
%
% Usage:
% [x, xexpt] = mrf_gibbs_mx(unary, pairwise, lbcost, gridsz, initx[, extraunary])
%
% --- Inputs ---
% unary: N*D matrix, the unary activation function for each element
% pairwise: N*(D*D) sparse matrix, each row is a flat sparse matrix specifying
%       pairwise terms for a data case. Pairwise potential
%       f_ij(x_i,x_j) = lbcost(x_i,x_j)*pairwise(i,j)
% lbcost: label cost, see above.
% gridsz: size of the grid, should satisfy that prod(gridsz) = D.
% initx: N*D matrix, initial state of the MRF.
% extraunary: extra unary potentials specified using a N*D matrix for x=1, i.e.
%       u(2,:) will be left unchanged, and u(1,:) will be added by 
%       extraunary(n,:).
%
% --- Output ---
% x: states of the model after one full block Gibbs iteration.
%
% Yujia Li, 05/2012
%

x = initx;
xexpt = initx;

mask = mrf_mask_gen(gridsz);

[numcases, numdims] = size(x);

xunaryact = unary;
if nargin > 5
    xunaryact = xunaryact + extraunary;
end

xpairact = zeros(numcases, numdims);

% sample the first part

for i = 1 : numcases
    % xpairact(i,:) = pairwise{i} * (lbcost(1,2-x(i,:)) - lbcost(2,2-x(i,:)))';
    xpairact(i,:) = pairwise{i} * (lbcost(2,x(i,:)+1) - lbcost(1,x(i,:)+1))';
end

xexpt(:,mask{1}) = 1 ./ (1+exp(-xunaryact(:,mask{1}) + xpairact(:,mask{1})));
x(:,mask{1}) = double(rand(numcases, sum(mask{1})) < xexpt(:,mask{1}));

% sample the second part

for i = 1 : numcases
    % xpairact(i,:) = pairwise{i} * (lbcost(1,2-x(i,:)) - lbcost(2,2-x(i,:)))';
    xpairact(i,:) = pairwise{i} * (lbcost(2,x(i,:)+1) - lbcost(1,x(i,:)+1))';
end

xexpt(:,mask{2}) = 1 ./ (1+exp(-xunaryact(:,mask{2}) + xpairact(:,mask{2})));
x(:,mask{2}) = double(rand(numcases, sum(mask{2})) < xexpt(:,mask{2}));

return
end

