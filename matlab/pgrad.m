function [x, fvals] = pgrad(fx, gx, x0, info)
% Apply adaptively preconditioned gradient method to strongly convex
% problems

x = x0;
d = info.D;
maxit = info.maxit;

fvals = zeros(maxit, 1);

for i = 1:maxit
    
    g = gx(x);
    f = fx(x);
    fvals(i) = f;
    x = x - d .* g;
    
    if norm(g) < info.tol
        break;
    end % End if 
        
end % End for


end % End function