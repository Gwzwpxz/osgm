from bench_algo import *
from algo_config import *
import numpy as np
from math import inf
from typing import Dict, Any

class GradientDescentLS(Optimizer):
    
    def __init__(self, params: dict = None):
        """
        Constructor for the gradient descent optimizer with Armijo line search.
        """
        if params is None:
            params = {}
            
        self.stats = {}
        
        super().__init__(params[ALG_UNIVERSAL_PARAM_NAME], params)
        
    def optimize(self, x: np.ndarray, f: callable, grad_f: callable) -> Dict[str, Any]:
        """
        Optimize the function f using Armijo line search.
        
        Parameters
        ----------
        x : np.ndarray
            Initial point
        f : callable
            Function to optimize
        grad_f : callable
            Gradient of the function
        
        Returns
        -------
        Dict[str, Any]
            Statistics of the optimizer
        """
        
        tol = self.params.get(ALG_UNIVERSAL_PARAM_TOL, 1e-06)
        max_iter = self.params.get(ALG_UNIVERSAL_PARAM_MAXITER, 1000)
        L_est = self.params.get(ALG_UNIVERSAL_PARAM_L_EST, inf)
        
        # Initial step size: alpha = 1 / L if L != inf, else 1e-3
        if L_est == inf:
            alpha_init = 1e-03
        else:
            alpha_init = 1.0 / L_est
        
        # Armijo parameter 
        c = 1e-04
        
        # For mild expansion if no backtracking occurred in the previous iteration
        expansion_factor = 1.2
        backtrack_steps_previous = 0

        # Counters
        n_func_evals = 0
        n_grad_evals = 0
        n_iter = 0
        
        # Statistics
        fvals  = np.zeros(max_iter)
        gnorms = np.zeros(max_iter)
        
        for i in range(max_iter):
            # Evaluate gradient & function
            gx = grad_f(x)
            n_grad_evals += 1
            
            fx = f(x)
            n_func_evals += 1
            
            fvals[i] = fx
            grad_norm = np.linalg.norm(gx, ord=np.inf)
            gnorms[i] = grad_norm
            n_iter += 1
            
            # Check stopping criterion
            if grad_norm < tol:
                break
            
            # Possibly enlarge alpha if no backtracking was needed in the previous iteration
            if i > 0 and backtrack_steps_previous == 0:
                alpha_init *= expansion_factor
            
            # Reset alpha for this iteration
            alpha = alpha_init
            backtrack_steps = 0
            
            # Armijo line search loop
            while True:
                x_new = x - alpha * gx
                fx_new = f(x_new)
                n_func_evals += 1 
                
                # Armijo condition: f(x_new) <= f(x) - c * alpha * ||gx||^2
                if fx_new <= fx - c * alpha * grad_norm ** 2:
                    x = x_new
                    break
                else:
                    alpha *= 0.5
                    backtrack_steps += 1
            
            # Keep track of the final alpha used this iteration
            alpha_init = alpha
            backtrack_steps_previous = backtrack_steps
        
        # If we stopped early, fill the tail of arrays with the last recorded values
        if n_iter < max_iter:
            fvals[n_iter:]  = fvals[n_iter - 1]
            gnorms[n_iter:] = gnorms[n_iter - 1]
            
        stats = {
            ALG_STATS_ITERATIONS: n_iter,
            ALG_STATS_OPTIMAL_VALUE: f(x),
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
        Get the optimizer statistics.
        
        Returns
        -------
        Dict[str, Any]
            Statistics of the optimizer
        """
        return self.stats
        
        
if __name__ == "__main__":
    
    # Example usage
    params = gradient_descent_params
    params[ALG_UNIVERSAL_PARAM_L_EST] = 4.0
    
    # Initialize the line search optimizer
    gd_ls = GradientDescentLS(params)
    
    # Initial guess
    x_init = np.array([0.0, 0.0])
    
    # Run optimization
    stats = gd_ls.optimize(x_init, f, grad_f)
    
    # Print results
    print("Optimizer Stats (Armijo line search):")
    print(f"Iterations:         {stats[ALG_STATS_ITERATIONS]}")
    print(f"Optimal Value:      {stats[ALG_STATS_OPTIMAL_VALUE]:.6f}")
    print(f"Optimal Solution:   {stats[ALG_STATS_OPTIMAL_SOL]}")
    print(f"Function Evaluations: {stats[ALG_STATS_FEVALS]}")
    print(f"Gradient Evaluations: {stats[ALG_STATS_GEVALS]}")
    print(f"Final Gradient Norm: {np.linalg.norm(grad_f(stats[ALG_STATS_OPTIMAL_SOL])):.6f}")
    print("\nFunction Values (first 5):", stats[ALG_STATS_FUNCVALS][:5])
    print("Gradient Norms (first 5):", stats[ALG_STATS_GNORMS][:5])
    print("\nTest completed!")
