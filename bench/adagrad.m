function [x, fvals] = adagrad(fx, gx, x0, info)
% Apply adaptively preconditioned gradient method to strongly convex
% problems

maxit = info.maxit;
alpha = info.adagradalpha;

x = x0;
n = length(x);

fvals = zeros(maxit, 1);
fvals(1) = fx(x);

G = zeros(n, 1);

for i = 1:maxit + 1
    
    g = gx(x);
    f = fx(x);
    
    fvals(i + 1) = f;
   
    G = G + g.^2;     
    x = x - alpha * g ./ sqrt(G + 1e-12);
    
    if norm(g) < info.tol
        break;
    end % End if 
    
end % End for

end % End function