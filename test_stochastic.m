clear; clc; close all;

addpath(genpath("."));
try
    cvx_solver;
catch
    error("CVX not installed");
end % End try

rng(123);
m = 100;
n = 100;
A = randn(n, n);
A = A * 0.01 + eye(m);
D = diag(rand(n, 1));
A = A * D * A';
sigma = 1e-03;
A = A + eye(n) * sigma;
b = rand(n, 1);

ATA = A' * A;
ATb = A' * b;

invA = inv(A' * A);
invA = (invA + invA') / 2;

cvx_begin sdp quiet
cvx_solver sdpt3
variable dopt(n)
variable tau
maximize( tau )
subject to
diag(dopt) <= invA;
diag(dopt) >= tau * invA;
cvx_end

L = eigs(ATA, 1, 'largestabs');
mu = eigs(ATA, 1, 'smallestabs');

fx = @(x) (0.5 * norm(A * x - b)^2);
gx = @(x) ATA * x - ATb;

stoch_fx = @(x, idx) (0.5 * norm(A(idx,:) * x - b(idx))^2);
stoch_gx = @(x, idx) A(idx,:)' * (A(idx,:) * x - b(idx));

x0 = randn(n, 1);
x0 = x0 / norm(x0);
info.L = L;
info.mu = mu;
info.tol = 1e-10;
info.z = 0;
info.P0 = [];
info.idiag = 1;
info.maxit = 10000;
info.adagradalpha = 10;
info.Hess = ATA;
info.batch_size = 10;
info.sgd_step_size = 1;

% Stochastic algorithms
[xgd, fvalsgd] = sgd(stoch_fx, stoch_gx, x0, info);
[xrx, frx] = osgmrx_stochastic(stoch_fx, stoch_gx, x0, info);
[xhx, fhx] = osgmhx_stochastic(stoch_fx, stoch_gx, x0, info);
[xgx, fgx] = osgmgx_stochastic(stoch_fx, stoch_gx, x0, info);

function smoothed_data = moving_average(data, window_size)
    smoothed_data = filter(ones(1, window_size)/window_size, 1, data);
end
window_size = 200;
fvalsgd_smooth = moving_average(fvalsgd, window_size);
frx_smooth = moving_average(frx, window_size);
fgx_smooth = moving_average(fgx, window_size);
fhx_smooth = moving_average(fhx, window_size);


% Plotting
linewid = 3;

semilogy(fvalsgd_smooth, 'LineWidth', linewid, 'DisplayName', 'SGD');
hold on;
semilogy(frx_smooth, 'LineWidth', linewid, 'DisplayName', 'OSGM-R');
semilogy(fgx_smooth, 'LineWidth', linewid, 'DisplayName', 'OSGM-G');
semilogy(fhx_smooth, 'LineWidth', linewid, 'DisplayName', 'OSGM-H');

legend();
set(gcf,'Position',[200 200 600 400])

xlim([1, info.maxit]);
grid on;

set(gca, 'FontSize', 20, 'LineWidth', 1, 'Box', 'on');
title(sprintf("$\\sigma = %5.4f \\quad \\kappa = %5.2f \\quad \\kappa^\\star = %5.2f$",...
        sigma, cond(ATA), 1 / tau), 'Interpreter', 'latex');
