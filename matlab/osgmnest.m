function [x, fvals] = osgmnest(fx, gx, x0, info)
% Online scaled gradient method with Nesterov acceleration

L = info.L;
mu = info.mu;
maxit = info.maxit;

Q = mu / L;
sqrtQ = sqrt(Q);

x = x0;
z = x0;

fvals = zeros(maxit, 1);

n = length(x);    

P = info.P0;
adagradalpha = info.adagradalpha;

if isempty(P)
    P = zeros(n, 1);
end % End if
Pv = @(P, g) P .* g;
G = zeros(n, 1);

for i = 1:maxit
    
    fvals(i) = fx(x);
    
    y = x + sqrtQ / (1 + sqrtQ) * (z - x);
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
    end % End if
    
    z = (1 - sqrtQ) * z + sqrtQ * (y - 1/mu * g);
   
    nrmg = norm(gx(x));
    if nrmg < info.tol
        break;
    end % End if
    
end % end for

end % End function