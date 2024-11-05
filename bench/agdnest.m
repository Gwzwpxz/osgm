function [x, fvals] = agdnest(fx, gx, x0, info)
% Implement Nesterov acceleration for strongly convex optimization

L = info.L;
mu = info.mu;
maxit = info.maxit;

Q = L / mu;
sqrtQ = sqrt(Q);
theta = (sqrtQ - 1) / (sqrtQ + 1);

x = x0;
y = x0;

fvals = zeros(maxit, 1);

for i = 1:maxit
    
    fvals(i) = fx(x);
    g = gx(x);
    
    ynew = x - (1/L) * g;
    x = (1 + theta) * ynew - theta * y;
    y = ynew;
    
    nrmg = norm(gx(y));
    
    if nrmg < info.tol
        break;
    end % End if
    
end % end for

end % End function