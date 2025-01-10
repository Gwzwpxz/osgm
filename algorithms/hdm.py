from bench_algo import *
from algo_config import *
import numpy as np
from scipy.stats import hmean, gmean

class HyperGradientDescent(Optimizer):
    """
    Implement hypergradient descent with momentum
    Optimized version with reduced function and gradient calls
    
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Constructor for the hypergradient descent optimizer.
        """
        
        if params is None:
            params = {}
        
        self.stats = {}
        super().__init__(params.get(ALG_UNIVERSAL_PARAM_NAME, "AdaGrad"), params)
        
    def optimize(self, x: np.ndarray, f: callable, grad_f: callable) -> Dict[str, Any]:
        """
        Optimize the function f using hypergradient descent with momentum
        
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
            A dictionary containing optimization statistics
        
        """
        
        # Extract parameters
        tol = self.params.get(ALG_UNIVERSAL_PARAM_TOL, 1e-06)
        max_iter = self.params.get(ALG_UNIVERSAL_PARAM_MAXITER, 1000)
        version = self.params.get(ALG_HDM_VERSION, ALG_HDM_VERSION_DIAG)
        lr = self.params.get(ALG_HDM_LEARNING_RATE, -1)
        beta_lr = self.params.get(ALG_HDM_BETA_LEARNING_RATE, 1.0)
        L_est = self.params.get(ALG_UNIVERSAL_PARAM_L_EST, np.inf)
        L_guesses = []
        
        is_learned = 0
        is_guessed = 0
        n_monotone_step = 0
        
        if L_est != np.inf and lr == -1:
            lr = 1.0 / L_est
        
        if lr == -1:
            lr = 0.1
        
        G = None
        
        # Scaling matrix learning buffer
        if version == ALG_HDM_VERSION_DIAG:
            G = np.zeros_like(x) 
            P = np.zeros_like(x)
        elif version == ALG_HDM_VERSION_MATRIX:
            G = np.zeros((x.shape[0], x.shape[0]))
            P = np.zeros((x.shape[0], x.shape[0]))
        elif version == ALG_HDM_VERSION_SCALAR:
            G = 0.0
            P = 0.0
        else:
            raise ValueError("Unknown version of hypergradient descent")
        
        # Momentum learning buffer
        beta = 0.95
        Gm = 0.0
        
        # Counters
        n_func_evals = 0
        n_grad_evals = 0
        n_iter = 0

        # Statistics
        fvals = np.zeros(max_iter)
        gnorms = np.zeros(max_iter)
        
        x_old = x
        
        gx = grad_f(x)
        n_grad_evals += 1
        fx = f(x)
        gtmp = np.zeros_like(gx)
        ftmp = 0.0
        
        gnorm_eps = 1e-20
        adagrad_eps = 1e-12
        
        for i in range(max_iter):
        
            grad_norm = np.linalg.norm(gx)
            grad_norm_inf = np.linalg.norm(gx, ord=np.inf)
            n_iter += 1
            
            # Save info for stats
            fvals[i] = fx  # function value
            gnorms[i] = grad_norm_inf
            
            # print("%d: f=%f, |g|_inf=%f, beta=%f, lr=%f" % (i, fx, grad_norm_inf, beta, lr))
            
            # Check stopping condition
            if grad_norm_inf < tol:
                break
            
            # Update the primal iterate
            if version == ALG_HDM_VERSION_MATRIX:
                xtmp = x - P.dot(gx) + beta * (x - x_old)
            else:
                xtmp = x - P * gx + beta * (x - x_old)
                
            ftmp = f(xtmp)
            n_func_evals += 1
            gtmp = grad_f(xtmp)
            n_grad_evals += 1
            
            gnorm_eps = np.linalg.norm(x - x_old) ** 2 * L_est
            
            if version == ALG_HDM_VERSION_DIAG:
                gr = - gtmp * gx / (grad_norm ** 2 + gnorm_eps)
                G += gr ** 2
                P -= lr * gr / (np.sqrt(G) + adagrad_eps)
            elif version == ALG_HDM_VERSION_MATRIX:
                gr = - np.outer(gtmp, gx) / (grad_norm ** 2 + gnorm_eps)
                G += gr ** 2
                P -= lr * gr / (np.sqrt(G) + adagrad_eps)
            elif version == ALG_HDM_VERSION_SCALAR:
                gr = - np.dot(gtmp, gx) / (grad_norm ** 2 + gnorm_eps)
                G += gr ** 2
                P -= lr * gr / (np.sqrt(G) + adagrad_eps)
            else:
                raise ValueError("Unknown version of hypergradient descent")

            gm = np.dot(gtmp, x - x_old) / (grad_norm ** 2 + 1e-12)
            Gm += gm ** 2
            beta = beta - beta_lr * gm / (np.sqrt(Gm) + adagrad_eps)
            beta = min(max(beta, -1.0), 1.0)
            
            if is_learned:
                x_old = x
                x = xtmp
                fx = ftmp
                gx = gtmp
            else:
                if ftmp < fx:
                    # Curvature estimate
                    curve_est = np.linalg.norm(gtmp - gx) / np.linalg.norm(xtmp - x) 
                    # curve_est = 0.5 * (grad_norm ** 2) / (fx - ftmp)
                    L_guesses.append(curve_est)
                    x_old = x
                    x = xtmp
                    fx = ftmp
                    gx = gtmp
                    n_monotone_step += 1
                else:
                    n_monotone_step -= 2
                    n_monotone_step = max(0, n_monotone_step)
                    
            if n_monotone_step > 50 and not is_guessed:
                
                L_guess = gmean(L_guesses)
                # Average with initial estimate
                # L_guess = (L_guess ** 0.8) * (L_est ** 0.2)
                lr = 1.0 / L_guess
                G *= 0
                # P *= 0
                is_guessed = 1
                
        # If we ended early, fill trailing stats
        if n_iter < max_iter:
            fvals[n_iter:] = fvals[n_iter - 1]
            gnorms[n_iter:] = gnorms[n_iter - 1]
            
        # Collect stats
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
        Get the optimizer statistics
        
        Returns
        -------
        Dict[str, Any]
            Statistics of the optimizer
        """
        return self.stats
                
if __name__ == "__main__":
    
    params = hdm_params
    params[ALG_HDM_VERSION] = ALG_HDM_VERSION_DIAG
    params[ALG_HDM_LEARNING_RATE] = 0.25
    params[ALG_UNIVERSAL_PARAM_L_EST] = 4.0
    
    # Initialize the optimizer
    hdm = HyperGradientDescent(params)
    
    # Initial guess
    x_init = np.array([0.0, 0.0])
    
    # Run optimization
    stats = hdm.optimize(x_init, f, grad_f)
    
    # Print results
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