function [x, fvals] = mmtm(fx, gx, x0, info)
% Apply adaptively preconditioned gradient method with Polyak momentum 
% to strongly convex problems

x = x0;
xold = x;
d = info.D;
maxit = info.maxit;

fvals = zeros(maxit, 1);
beta = 0.995;

for i = 1:maxit
    
    g = gx(x);
    f = fx(x);
    fvals(i) = f;
    
    xtmp = x;
    x = x - d .* g + beta * (x - xold);
    
    if norm(g) < info.tol
        break;
    end % End if 
    
    xold = xtmp;
        
end % End for


end % End function