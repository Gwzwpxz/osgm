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

sigmas = [1e-4, 1e-3, 1e-2, 1e-1];
n_sigmas = length(sigmas);
window_size = 200;

figure;
t = tiledlayout(1, n_sigmas, 'TileSpacing', 'compact', 'Padding', 'compact');
linewid = 3;

function smoothed_data = moving_average(data, window_size)
    smoothed_data = filter(ones(1, window_size)/window_size, 1, data);
end

for i = 1:n_sigmas
    sigma = sigmas(i);
    A_sigma = A + eye(n) * sigma;
    b = rand(n, 1);

    ATA = A_sigma' * A_sigma;
    ATb = A_sigma' * b;

    invA = inv(A_sigma' * A_sigma);
    invA = (invA + invA') / 2;

    % Solve using CVX
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

    fx = @(x) (0.5 * norm(A_sigma * x - b)^2);
    gx = @(x) ATA * x - ATb;

    stoch_fx = @(x, idx) (0.5 * norm(A_sigma(idx,:) * x - b(idx))^2);
    stoch_gx = @(x, idx) A_sigma(idx,:)' * (A_sigma(idx,:) * x - b(idx));

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

    % Smooth data for plotting
    fvalsgd_smooth = moving_average(fvalsgd, window_size);
    frx_smooth = moving_average(frx, window_size);
    fgx_smooth = moving_average(fgx, window_size);
    fhx_smooth = moving_average(fhx, window_size);

    nexttile;
    semilogy(fvalsgd_smooth, 'LineWidth', linewid, 'DisplayName', 'SGD');
    hold on;
    semilogy(frx_smooth, 'LineWidth', linewid, 'DisplayName', 'OSGM-R');
    semilogy(fgx_smooth, 'LineWidth', linewid, 'DisplayName', 'OSGM-G');
    semilogy(fhx_smooth, 'LineWidth', linewid, 'DisplayName', 'OSGM-H');

    title(sprintf("$\\sigma = %5.4f \\quad \\kappa = %5.2f \\quad \\kappa^\\star = %5.2f$",...
        sigma, cond(ATA), 1 / tau), 'Interpreter', 'latex');
    xlim([1, length(fvalsgd_smooth)]);
    grid on;
    set(gca, 'FontSize', 14, 'LineWidth', 1, 'Box', 'on');
end

lgd = legend(t.Children(1), 'SGD', 'OSGM-R', 'OSGM-G', 'OSGM-H', 'Location', 'eastoutside');
set(lgd, 'FontSize', 12);

set(gcf, 'Position', [0, 0, 1800, 400]);

% saveas(gcf, 'stochastic_algorithms_plot.pdf');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0 0 18 4]);
set(gcf, 'PaperSize', [18 4]);

print('stochastic_algorithms_plot', '-dpdf', '-r300');
