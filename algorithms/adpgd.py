from bench_algo import *
from algo_config import *
import numpy as np
from math import inf, sqrt
from typing import Dict, Any

class AdpGD(Optimizer):
    """
    Adaptive gradient descent based on the local smoothness constant,
    as in the original Adgd(Trainer) logic.
    
    The update rules are:
        1. On the first iteration, set: 
           theta = inf, lr = LR0, w_old = x, grad_old = grad_f(x).
           Then do x <- x - lr * grad_old.
        2. Each iteration k:
           - Compute grad = grad_f(x).
           - Estimate local Lipschitz L = ||grad - grad_old|| / ||x - w_old||.
           - If theta = inf: lr_new = 0.5 / L.
             Else: lr_new = min(sqrt(1 + theta)*lr, EPS / lr + 0.5 / L).
           - Then: theta = lr_new / lr, lr = lr_new.
           - step: w_old = x, grad_old = grad, x <- x - lr * grad.
    """

    def __init__(self, params: dict = None):
        """
        Constructor for Adgd optimizer.
        """
        if params is None:
            params = {}
        self.stats = {}
        
        # Call base optimizer constructor
        super().__init__(params.get(ALG_UNIVERSAL_PARAM_NAME, "AdpGD"), params)

    def optimize(self, x: np.ndarray, f: callable, grad_f: callable) -> Dict[str, Any]:
        """
        Execute the AdpGD optimization loop.

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
            A dictionary of statistics (iterations, function values, etc.)
        """
        
        tol = self.params.get(ALG_UNIVERSAL_PARAM_TOL, 1e-06)
        max_iter = self.params.get(ALG_UNIVERSAL_PARAM_MAXITER, 1000)
        
        eps = self.params.get(ALG_ADP_GD_EPS, 0.0)
        if eps < 0.0:
            raise ValueError("Invalid eps: must be >= 0, got {}".format(eps))
        
        lr0 = self.params.get(ALG_ADP_GD_LR0, 1e-10)

        # --- Counters & storage ---
        n_func_evals = 0
        n_grad_evals = 0
        fvals = np.zeros(max_iter)
        gnorms = np.zeros(max_iter)

        # --- INITIALIZATION PHASE ---
        # Evaluate initial gradient and function value
        grad = grad_f(x)
        n_grad_evals += 1
        fx = f(x)
        n_func_evals += 1

        # Store statistics
        fvals[0] = fx
        gnorms[0] = np.linalg.norm(grad, ord=2)
        
        # Initialize step-size-related variables
        self.theta = np.inf  # signals first iteration
        self.lr = lr0
        
        # Save old references
        w_old = x.copy()
        grad_old = grad.copy()
        
        # First step
        x = x - self.lr * grad
        
        n_iter = 1
        
        for i in range(1, max_iter):
            # Compute new gradient at x
            grad = grad_f(x)
            n_grad_evals += 1
            fx = f(x)
            n_func_evals += 1

            grad_norm = np.linalg.norm(grad, ord=2)
            fvals[i]  = fx
            gnorms[i] = grad_norm
            n_iter += 1
            
            # Stopping criterion
            if grad_norm < tol:
                break
            
            # Estimate local Lipschitz:
            # L = ||grad - grad_old|| / ||x - w_old||
            diff_x = x - w_old
            norm_diff_x = np.linalg.norm(diff_x, ord=2)
            if norm_diff_x > 1e-14:
                diff_g = grad - grad_old
                L = np.linalg.norm(diff_g, ord=2) / norm_diff_x
            else:
                L = 1e+12 

            # Update step size
            if np.isinf(self.theta):
                lr_new = 0.5 / L
            else:
                lr_new = min(sqrt(1 + self.theta) * self.lr,
                             eps / self.lr + 0.5 / L)
            
            self.theta = lr_new / self.lr
            self.lr = lr_new

            # "Step": shift old variables and update x
            w_old = x.copy()
            grad_old = grad.copy()
            x = x - self.lr * grad
        
        # Fill any leftover array spots with final value
        if n_iter < max_iter:
            fvals[n_iter:]  = fvals[n_iter - 1]
            gnorms[n_iter:] = gnorms[n_iter - 1]

        # Store final statistics
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
    # Example usage:

    params = adpgd_params  # or an empty dict: {}
    params[ALG_ADP_GD_EPS] = 0.0    # default in original code
    params[ALG_ADP_GD_LR0] = 1e-03   # example initial step size
    
    # Initialize the ADGD optimizer
    adgd = AdpGD(params)
    
    # Initial point
    x_init = np.array([0.0, 0.0])
    
    # Run optimization
    stats = adgd.optimize(x_init, f, grad_f)
    
    # Print results
    print("AdpGD Optimizer Stats:")
    print(f"Iterations:          {stats[ALG_STATS_ITERATIONS]}")
    print(f"Optimal Value:       {stats[ALG_STATS_OPTIMAL_VALUE]:.6f}")
    print(f"Optimal Solution:    {stats[ALG_STATS_OPTIMAL_SOL]}")
    print(f"Function Evaluations: {stats[ALG_STATS_FEVALS]}")
    print(f"Gradient Evaluations: {stats[ALG_STATS_GEVALS]}")
    print(f"Final Gradient Norm: {np.linalg.norm(grad_f(stats[ALG_STATS_OPTIMAL_SOL])):.6e}")
    print("\nFunction Values (first 5):", stats[ALG_STATS_FUNCVALS][:5])
    print("Gradient Norms (first 5):", stats[ALG_STATS_GNORMS][:5])
    print("\nTest completed!")
