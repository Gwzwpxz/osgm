function [x, fvals] = osgmhxm(fx, gx, x0, info)
% Online scaled gradient method with hypergradient surrogate
% Wenzhi Gao, Stanford University
%
%  Input:   
%     fx: function value oracle
%     gx: gradient oracle
%     x0: initial point
%   info: other information
%         maxit: maximum iteration
%         idiag: diagonal sparsity pattern
%         P0: initial scaling matrix. Start from 0 if not specified
%         adagradalpha: stepsize in AdaGrad
%         tol: tolerance of gradient norm
% Output:
%      x: last solution
%  fvals: objectives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maxit = info.maxit;
idiag = info.idiag;
P = info.P0;
adagradalpha = info.adagradalpha;

x = x0;
n = length(x);    

fvals = zeros(maxit, 1);
fvals(1) = fx(x);

% Scaling matrix and vector multiplication
if idiag
    if isempty(P)
        P = zeros(n, 1);
    end % End if
    Pv = @(P, g) P .* g;
else    
    if isempty(P)
        P = zeros(n, n);
    end % End if
    Pv = @(P, g) P * g;
end % End if

% AdaGrad as online algorithm
if idiag 
    G = zeros(n, 1);
else
    G = zeros(n, n);
end % End if

ngradeval = 0;

% Initialize momentum parameter
beta = 0.0;
% Also use AdaGrad as online algorithm
Gm = 0;
xold = x;

for i = 1:maxit + 1
    
   g = gx(x);
   f = fx(x);
   nrmg = norm(g);
   
   fvals(i + 1) = f;
   
   xtmp = x - Pv(P, g) + beta .* (x - xold);
   ftmp = fx(xtmp);
   gtmp = gx(xtmp);     
   
   if idiag
       % Diagonal update
       gr = - (gtmp .* g) / nrmg^2;
       G = G + gr.^2;
       P = P - adagradalpha * gr ./ sqrt(G + 1e-20);
   else
       % Full matrix update
       gr = - (gtmp * g') / nrmg^2;
       G = G + gr.^2;
       P = P - adagradalpha * gr ./ sqrt(G + 1e-20);
   end % End if 
   
   % Momentum update
   gm = (gtmp' * (x - xold)) / nrmg^2;
   Gm = Gm + gm.^2;
   % betas(i) = beta;
   beta = beta - 100 * gm ./ sqrt(Gm + 1e-20);
   beta = min(beta, 1.0);
   beta = max(beta, -1.0);
   
   xold = x;
   
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