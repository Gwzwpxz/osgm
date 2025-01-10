function [x, fvals] = osgmhxmirror(fx, gx, x0, info)
% Online scaled gradient method with hypergradient surrogate and mirror
% descent update
% Wenzhi Gao, Stanford University
%
%  Input:   
%     fx: function value oracle
%     gx: gradient oracle
%     x0: initial point
%   info: other information
%         maxit: maximum iteration
%         P0: initial scaling matrix. Start from 0 if not specified
%         tol: tolerance of gradient norm
%         eta: online mirror descent stepsize scalar
%         dlt: simplex size
% Output:
%      x: last solution
%  fvals: objectives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maxit = info.maxit;
P = info.P0;
eta = info.eta;
dlt = info.dlt;

x = x0;
n = length(x); 

fvals = zeros(maxit, 1);
fvals(1) = fx(x);

% Scaling matrix and vector multiplication
if isempty(P)
    P = dlt * ones(n, 1) / n;
end % End if

assert(size(P, 1) == n);
assert(size(P, 2) == 1);

Pv = @(P, g) P .* g;
ngradeval = 0;

for i = 1:maxit + 1
    
   g = gx(x);
   f = fx(x);
   nrmg = norm(g);
   
   fvals(i + 1) = f;
   
   xtmp = x - Pv(P, g);
   ftmp = fx(xtmp);
   gtmp = gx(xtmp);   
   
   % Diagonal update
   gr = - (gtmp .* g) / norm(g)^2;
   P = P .* exp(- eta * gr / sqrt(i));
   
   if sum(P) > dlt
       P = dlt * P / sum(P);
   end % End if
   
   % Monotone oracle
   if ftmp < f
       x = xtmp;
       ngradeval = ngradeval + 1;
   else
       ngradeval = ngradeval + 2;
   end % End if
   
   if nrmg < info.tol
       break;
   end % End if
       
end % End for

end % End function