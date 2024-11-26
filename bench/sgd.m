function [x, fvals] = sgd(fx, gx, x0, info)
% Simple implementation of Stochastic Gradient Descent (SGD)
%
%  Input:
%     fx: stochastic function value oracle
%     gx: stochastic gradient oracle
%     x0: initial point
%   info: additional parameters
%         maxit: maximum iterations
%         batch_size: mini-batch size
%         step_size: learning rate
%         tol: tolerance for gradient norm
% Output:
%      x: last solution
%  fvals: objectives over iterations

% Unpack info
maxit = info.maxit;
batch_size = info.batch_size;
step_size = info.sgd_step_size;
tol = info.tol;

x = x0;
n = length(x);
fvals = zeros(maxit + 1, 1);  % +1 to account for initial value
fvals(1) = fx(x, 1:n);  % Evaluate full function initially

for i = 1:maxit
    % Sample a random mini-batch
    idx = randperm(n, batch_size);
    
    % Compute stochastic gradient and function value
    g = gx(x, idx);
    f = fx(x, idx);
    fvals(i + 1) = f;
    
    % Update step
    x = x - step_size * g;
    
    % Check for convergence
    if norm(g) < tol
        fvals = fvals(1:i + 1); % Include current iteration
        break;
    end
end

end
