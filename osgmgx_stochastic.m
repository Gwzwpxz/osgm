function [x, fvals] = osgmgx_stochastic(fx, gx, x0, info)
% Online scaled gradient method with gradient norm surrogate. Only for fixed Hessian (Stochastic version)
% Herman Brunborg & Wenzhi Gao, Stanford University
%
%  Input:
%     fx: stochastic function value oracle
%     gx: stochastic gradient oracle
%     x0: initial point
%   info: other information
%         maxit: maximum iteration
%         idiag: diagonal sparsity pattern
%         P0: initial scaling matrix. Start from 0 if not specified
%         adagradalpha: stepsize in AdaGrad
%         tol: tolerance of gradient norm
%         Hess: fixed Hessian matrix
%         batch_size: size of mini-batch for stochastic sampling
% Output:
%      x: last solution
%  fvals: objectives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maxit = info.maxit;
idiag = info.idiag;
P = info.P0;
% Hessian is considered fixed
H = info.Hess;
adagradalpha = info.adagradalpha;
batch_size = info.batch_size;

x = x0;
n = length(x);

fvals = zeros(maxit, 1);
fvals(1) = fx(x, 1:n);  % Evaluate full function initially

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

% Adagrad as online algorithm
if idiag
    G = zeros(n, 1);
else
    G = zeros(n, n);
end % End if

ngradeval = 0;

for i = 1:maxit + 1
    % Sample mini-batch
    idx = randperm(n, batch_size);

    g = gx(x, idx);
    f = fx(x, idx);
    nrmg = norm(g);

    fvals(i + 1) = f;

    xtmp = x - Pv(P, g);
    gtmp = gx(xtmp, idx);

    if idiag
        % Diagonal update
        gr = - ((H * gtmp) .* g) / (norm(g) * norm(gtmp));
        G = G + gr.^2;
        P = P - adagradalpha * gr ./ sqrt(G + 1e-20);
    else
        % Full matrix update
        gr = - ((H * gtmp) * g') / (norm(g) * norm(gtmp));
        G = G + gr.^2;
        P = P - adagradalpha * gr ./ sqrt(G + 1e-20);
    end % End if

    % Monotone oracle
    if norm(gtmp) < nrmg
        x = xtmp;
    end % End if

    if nrmg < info.tol
        break;
    end % End if

end % End for

end % End function
