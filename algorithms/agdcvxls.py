from bench_algo import *
from algo_config import *
import numpy as np
from math import inf, sqrt
from typing import Dict, Any

class AcceleratedGradientCvxLS(Optimizer):
    """
    Implements Nesterov's Accelerated Gradient (NAG) for convex functions,
    but replaces the fixed step size with a backtracking (Armijo) line search.
    
    The main update is:
        1) (Momentum) lbd_{k+1} = (1 + sqrt(1 + 4 lbd_k^2)) / 2
           beta_k = (lbd_k - 1) / lbd_{k+1}
           y_k = x_k + beta_k * (x_k - x_{k-1})
        2) (Line search) x_{k+1} = y_k - alpha * grad_f(y_k),
           where alpha is found by Armijo backtracking around y_k.
    """

    def __init__(self, params: Dict[str, Any] = None):
        """
        Constructor for the accelerated gradient with line search.
        """
        if params is None:
            params = {}

        self.stats = {}
        
        # Rename the optimizer internally
        super().__init__(
            params.get(ALG_UNIVERSAL_PARAM_NAME, "AcceleratedGradientCvxLS"), 
            params
        )

    def optimize(self, x: np.ndarray, f: callable, grad_f: callable) -> Dict[str, Any]:
        """
        Optimize function f using Nesterov's accelerated gradient (with Armijo line search).
        
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
            Dictionary of optimization statistics
        """
        # --- 1) Extract parameters ---
        tol      = self.params.get(ALG_UNIVERSAL_PARAM_TOL, 1e-6)
        max_iter = self.params.get(ALG_UNIVERSAL_PARAM_MAXITER, 1000)
        L_est    = self.params.get(ALG_UNIVERSAL_PARAM_L_EST, np.inf)

        # If user did not provide L_est, we can default to a small alpha_init:
        if L_est == np.inf:
            alpha_init = 1e-03
        else:
            alpha_init = 1.0 / L_est
        
        # Armijo backtracking constants
        c = 1e-04
        expansion_factor = 1.2

        # Keep track of backtracking from previous iteration
        backtrack_steps_previous = 0

        # Counters
        n_func_evals = 0
        n_grad_evals = 0
        n_iter = 0

        # Statistics arrays
        fvals  = np.zeros(max_iter)
        gnorms = np.zeros(max_iter)

        # Momentum-related variables
        lbd = 0.0
        x_prev = np.copy(x)

        for i in range(max_iter):
            if i == 0:
                # First iteration: no momentum yet
                y = x
                lbd = 1.0
            else:
                lbdtmp = 0.5 * (1 + sqrt(1 + 4 * (lbd**2)))
                beta = (lbd - 1.0) / lbdtmp
                lbd = lbdtmp
                # Lookahead point
                y = x + beta * (x - x_prev)

            fx = f(x)
            n_func_evals += 1
            
            gx = grad_f(x)
            gy = grad_f(y)
            n_grad_evals += 1  # counting both calls

            # Record stats
            grad_norm = np.linalg.norm(gx, ord=np.inf)
            fvals[i]  = fx
            gnorms[i] = grad_norm
            n_iter += 1
            
            # Stopping criterion
            if grad_norm < tol:
                break

            if i > 0 and backtrack_steps_previous == 0:
                alpha_init *= expansion_factor
            
            alpha = alpha_init
            backtrack_steps = 0
            
            # Evaluate f(y) once (we already have this from above)
            fy = f(y)
            n_func_evals += 1

            # Armijo loop
            while True:
                
                # Proposed Nesterov step
                x_next = y - alpha * gy
                fx_next = f(x_next)
                n_func_evals += 1
                
                # Armijo condition: f(x_next) <= f(y) - c * alpha * ||gy||^2
                if fx_next <= fy - c * alpha * np.sum(gy * gy):
                    # Accept the step
                    break
                else:
                    alpha *= 0.5
                    backtrack_steps += 1
            
            alpha_init = alpha
            backtrack_steps_previous = backtrack_steps

            x_prev = np.copy(x)
            x = x_next  # Accept the new point

        # If we ended early, fill trailing stats
        if n_iter < max_iter:
            fvals[n_iter:] = fvals[n_iter - 1]
            gnorms[n_iter:] = gnorms[n_iter - 1]

        # Final objective at the solution
        fx_final = f(x)
        n_func_evals += 1

        # --- 5) Collect stats ---
        stats = {
            ALG_STATS_ITERATIONS: n_iter,
            ALG_STATS_OPTIMAL_VALUE: fx_final,
            ALG_STATS_OPTIMAL_SOL: x,
            ALG_STATS_RUNNING_TIME: 0,  # placeholder
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
    
    # Example usage
    params = accelerated_gradient_descent_cvx_params
    params[ALG_UNIVERSAL_PARAM_L_EST] = 4.0
    
    # Initialize the optimizer
    agdcvx = AcceleratedGradientCvxLS(params)
    
    # Initial guess
    x_init = np.array([0.0, 0.0])
    
    # Run optimization
    stats = agdcvx.optimize(x_init, f, grad_f)
    
    # Print results
    print("Optimizer Name:", params[ALG_UNIVERSAL_PARAM_NAME])
    print("Optimizer Stats:")
    print(f"Iterations:         {stats[ALG_STATS_ITERATIONS]}")
    print(f"Optimal Value:      {stats[ALG_STATS_OPTIMAL_VALUE]:.6f}")
    print(f"Optimal Solution:   {stats[ALG_STATS_OPTIMAL_SOL]}")
    print(f"Function Evaluations: {stats[ALG_STATS_FEVALS]}")
    print(f"Gradient Evaluations: {stats[ALG_STATS_GEVALS]}")
    print(f"Final Gradient Norm: {np.linalg.norm(grad_f(stats[ALG_STATS_OPTIMAL_SOL])):.6f}")
    print("\nFunction Values (first 5):", stats[ALG_STATS_FUNCVALS][0:5])
    print("Gradient Norms (first 5):", stats[ALG_STATS_GNORMS][0:5])
    print("\nTest completed!")
    
    