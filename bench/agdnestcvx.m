function [x, fvals] = agdnestcvx(fx, gx, x0, info)
% Implement Nesterov acceleration for general convex optimization

L = info.L;
maxit = info.maxit;

x = x0;
y = x0;

fvals = zeros(maxit, 1);

lbd = 0;

for i = 1:maxit
    
    fvals(i) = fx(x);
    g = gx(x);
    
    ynew = x - (1/L) * g;
    lbdtmp = (1 + sqrt(1 + 4 * lbd^2)) / 2;
    theta = (lbd - 1) / lbdtmp;
    lbd = lbdtmp;
    
    x = (1 + theta) * ynew - theta * y;
    y = ynew;
    
    nrmg = norm(gx(y));
    
    if nrmg < info.tol
        break;
    end % End if
    
end % end for

end % End function