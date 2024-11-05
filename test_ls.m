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

x0 = randn(n, 1);
x0 = x0 / norm(x0);
z = 0;
djacob = 1 ./ diag(ATA);

info.L = L;
info.mu = mu;
info.tol = 1e-10;
info.P0 = [];
info.z = 0;
info.idiag = 1;
info.D = 1 / L;
info.maxit = 10000;
info.adagradalpha = 10;
info.Hess = ATA;

[xgd, fvalsgd] = pgrad(fx, gx, x0, info);
info.D = dopt;
[xoptdiag, foptdiag] = pgrad(fx, gx, x0, info);
[xrx, frx] = osgmrx(fx, gx, x0, info);
[xhx, fhx] = osgmhx(fx, gx, x0, info);
[xnes, fnes] = agdnest(fx, gx, x0, info);
[xnescvx, fnescvx] = agdnestcvx(fx, gx, x0, info);
[xgx, fgx] = osgmgx(fx, gx, x0, info);
[xada, fada] = adagrad(fx, gx, x0, info);

linewid = 3;

semilogy(fvalsgd, 'LineWidth', linewid, 'DisplayName', 'GD');
hold on;
semilogy(foptdiag, 'LineWidth', linewid, 'DisplayName', 'OptDiag');
semilogy(frx, 'LineWidth', linewid, 'DisplayName', 'OSGM-R');
semilogy(fgx, 'LineWidth', linewid, 'DisplayName', 'OSGM-G');
semilogy(fhx, 'LineWidth', linewid, 'DisplayName', 'OSGM-H');
semilogy(fnes, 'LineWidth', linewid, 'DisplayName', 'SAGD');
semilogy(fnescvx, 'LineWidth', linewid, 'DisplayName', 'AGD');
semilogy(fada, 'LineWidth', linewid, 'DisplayName', 'AdaGrad', 'LineStyle', ':');
legend();

set(gcf,'Position',[200 200 600 400])

xlim([1, 2000]);
grid on;

set(gca, 'FontSize', 20, 'LineWidth', 1, 'Box', 'on');
title(sprintf("$\\sigma = %5.4f \\quad \\kappa = %5.2f \\quad \\kappa^\\star = %5.2f$",...
        sigma, cond(ATA), 1 / tau), 'Interpreter', 'latex');