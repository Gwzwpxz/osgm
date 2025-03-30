from bench_algo import *
from algo_config import *
import numpy as np
from numpy import linalg as la
from math import sqrt, inf
from typing import Dict, Any

class AdpGDAcc(Optimizer):
    """
    Adaptive gradient descent with a heuristic Nesterov-type acceleration.
    Targeted at locally strongly convex functions.

    The key per-iteration updates:
      1) Estimate local smoothness L  = ||grad - grad_old|| / ||x - x_old||.
      2) Estimate new learning rate   lr = min( sqrt(1 + a_lr*theta_lr)*lr, b_lr / L ).
         Then theta_lr = lr_new / lr.
      3) Estimate new "strong convexity" mu similarly:
         mu_new = min( sqrt(1 + a_mu*theta_mu)*mu, b_lr * L )  # from original code
         Then theta_mu = mu_new / mu.
      4) momentum = ( sqrt(1/lr) - sqrt(mu) ) / ( sqrt(1/lr) + sqrt(mu) )
         w_nesterov_{k+1} = x_k - lr * grad_k
         x_{k+1}          = w_nesterov_{k+1} + momentum * (w_nesterov_{k+1} - w_nesterov_k)

    """

    def __init__(self, params: dict = None):
        """
        Constructor for AdgdAccel optimizer.
        """
        if params is None:
            params = {}
        self.stats = {}
        
        # Call the base Optimizer constructor
        super().__init__(params.get(ALG_UNIVERSAL_PARAM_NAME, "AdgdAccel"), params)

    def optimize(self, x: np.ndarray, f: callable, grad_f: callable) -> Dict[str, Any]:
        """
        Run the AdgdAccel optimization loop.
        
        Parameters
        ----------
        x : np.ndarray
            Initial point
        f : callable
            Objective function
        grad_f : callable
            Gradient of the objective function
        
        Returns
        -------
        Dict[str, Any]
            Statistics of the optimizer (function values, gradient norms, etc.)
        """

        # Extract parameters
        tol      = self.params.get(ALG_UNIVERSAL_PARAM_TOL, 1e-6)
        max_iter = self.params.get(ALG_UNIVERSAL_PARAM_MAXITER, 1000)

        a_lr = self.params.get(ALG_ADP_GDACC_ALR, 0.5)    # a_lr
        a_mu = self.params.get(ALG_ADP_GDACC_AMU, 0.5)    # a_mu
        b_lr = self.params.get(ALG_ADP_GDACC_BLR, 0.5)    # b_lr
        b_mu = self.params.get(ALG_ADP_GDACC_BMU, 0.5)    # b_mu

        # Counters
        n_func_evals = 0
        n_grad_evals = 0
        fvals  = np.zeros(max_iter)
        gnorms = np.zeros(max_iter)

        # Evaluate initial gradient
        grad = grad_f(x)
        n_grad_evals += 1
        grad_norm = la.norm(grad, ord=2)
        
        # Heuristic initial step-size, from the original:
        #   lr = 1e-5 / la.norm(grad)
        # Also a "mu" = 1 / lr.
        if grad_norm < 1e-14:
            lr = 1e-03
        else:
            lr = 1e-05 / grad_norm

        mu = 1.0 / lr
        
        # Nesterov memory
        w_nesterov = x.copy()
        w_nesterov_old = w_nesterov.copy()
        
        # We'll keep old references to measure local Lipschitz
        x_old = x.copy()
        grad_old = grad.copy()

        # Theta trackers for the step-size and strong-convexity updates
        theta_lr = inf
        theta_mu = inf

        # First step: x <- x - lr * grad
        x = x - lr * grad
        fx = f(x)
        n_func_evals += 1

        fvals[0]  = fx
        gnorms[0] = grad_norm
        
        n_iter = 1

        for k in range(1, max_iter):
            # Evaluate gradient
            grad = grad_f(x)
            n_grad_evals += 1
            grad_norm = la.norm(grad, ord=2)
            
            fx = f(x)
            
            fvals[k]  = fx
            gnorms[k] = grad_norm
            n_iter += 1

            # --- Check stopping condition ---
            if grad_norm < tol:
                break
            
            # --- Estimate local Lipschitz: L = ||grad - grad_old|| / ||x - x_old||
            diff_x = x - x_old
            norm_diff_x = la.norm(diff_x, ord=2)
            if norm_diff_x > 1e-14:
                diff_g = grad - grad_old
                L = la.norm(diff_g, ord=2) / norm_diff_x
            else:
                L = 1e+12  # fallback if we didn't move

            # lr_new = min( sqrt(1 + a_lr * theta_lr)*lr, b_lr / L )
            lr_cand = sqrt(1 + a_lr * theta_lr) * lr
            lr_new = min(lr_cand, b_lr / (L if L > 1e-14 else 1e-14))
            theta_lr = lr_new / lr
            lr = lr_new

            # mu_new = min( sqrt(1 + a_mu * theta_mu)*mu, b_lr * L ) 
            mu_cand = sqrt(1 + a_mu * theta_mu) * mu
            mu_new = min(mu_cand, b_lr * L)
            theta_mu = mu_new / mu
            mu = mu_new

            # momentum = ( sqrt(1/lr) - sqrt(mu) ) / ( sqrt(1/lr) + sqrt(mu) )
            # w_nesterov_{k+1} = x_k - lr * grad
            # x_{k+1} = w_nesterov_{k+1} + momentum*( w_nesterov_{k+1} - w_nesterov_k )
            if lr < 1e-14:
                momentum = 0.0  # fallback if lr is extremely small
            else:
                momentum = (sqrt(1.0 / lr) - sqrt(mu)) / (sqrt(1.0 / lr) + sqrt(mu))

            w_nesterov_old = w_nesterov.copy()
            w_nesterov = x - lr * grad
            x_new = w_nesterov + momentum * (w_nesterov - w_nesterov_old)

            # Shift old references
            x_old = x.copy()
            grad_old = grad.copy()
            
            # Accept new x
            x = x_new

        if n_iter < max_iter:
            fvals[n_iter:]  = fvals[n_iter - 1]
            gnorms[n_iter:] = gnorms[n_iter - 1]
        
        stats = {
            ALG_STATS_ITERATIONS: n_iter,
            ALG_STATS_OPTIMAL_VALUE: fx,
            ALG_STATS_OPTIMAL_SOL: x,
            ALG_STATS_RUNNING_TIME: 0,
            ALG_STATS_FUNCVALS: fvals,
            ALG_STATS_GNORMS: gnorms,
            ALG_STATS_FEVALS: n_func_evals,
            ALG_STATS_GEVALS: n_grad_evals
        }
        
        self.stats = stats
        return stats
    
    def get_optimizer_stats(self) -> Dict[str, Any]:
        """
        Return the optimizer's collected statistics.
        """
        return self.stats


if __name__ == "__main__":

    params = {
        ALG_UNIVERSAL_PARAM_TOL: 1e-06,
        ALG_UNIVERSAL_PARAM_MAXITER: 1000,
        ALG_ADP_GDACC_ALR: 0.5,
        ALG_ADP_GDACC_BLR: 0.5,
        ALG_ADP_GDACC_AMU: 0.5,  
        ALG_ADP_GDACC_BMU: 0.5   
    }
    
    # Initialize the AdgdAccel optimizer
    adpaccgd = AdpGDAcc(params)

    # Initial guess
    x_init = np.array([0.0, 0.0])

    # Optimize
    stats = adpaccgd.optimize(x_init, f, grad_f)

    # Print results
    print("AdpGD Acc Optimizer Stats:")
    print(f"Iterations:           {stats[ALG_STATS_ITERATIONS]}")
    print(f"Optimal Value:        {stats[ALG_STATS_OPTIMAL_VALUE]:.6f}")
    print(f"Optimal Solution:     {stats[ALG_STATS_OPTIMAL_SOL]}")
    print(f"Function Evaluations: {stats[ALG_STATS_FEVALS]}")
    print(f"Gradient Evaluations: {stats[ALG_STATS_GEVALS]}")
    print(f"Final Gradient Norm:  {np.linalg.norm(grad_f(stats[ALG_STATS_OPTIMAL_SOL])):.6e}")
    print("\nFunction Values (first 5):", stats[ALG_STATS_FUNCVALS][:5])
    print("Gradient Norms (first 5):", stats[ALG_STATS_GNORMS][:5])
    print("Done!")
