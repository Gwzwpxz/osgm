function [x, fvals] = osgmnestcvx(fx, gx, x0, info)
% Implement Nesterov acceleration for general convex optimization

L = info.L;
maxit = info.maxit;

x = x0;
z = x0;

fvals = zeros(maxit, 1);
n = length(x);

Ak = 0;

P = info.P0;
adagradalpha = info.adagradalpha;

if isempty(P)
    P = zeros(n, 1);
end % End if
Pv = @(P, g) P .* g;
G = zeros(n, 1);

for i = 1:maxit
    
    fvals(i) = fx(x);
    ak = 0.5 * (1 + sqrt(4 * Ak + 1));
    
    y = x + (1 - Ak / (Ak + ak)) * (z - x);
    g = gx(y);
    
    x = y - (1/L) * g;
    nrmg = norm(g);
    xtmp = y - Pv(P, g);
    ftmp = fx(xtmp);
    gtmp = gx(xtmp);
    gr = - (gtmp .* g) / nrmg^2;
    G = G + gr.^2;
    P = P - adagradalpha * gr ./ sqrt(G + 1e-20);
    
    if ftmp <= fx(x)
        x = xtmp;
        hxP = (ftmp - fx(y)) / nrmg^2;
        z = z + 2 * ak * hxP * g;
    else
        z = z - ak / L * g;
    end % End if
    
    Ak = Ak + ak;
   
    nrmg = norm(gx(x));
    if nrmg < info.tol
        break;
    end % End if
    
end % end for

end % End function