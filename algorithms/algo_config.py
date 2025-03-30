from numpy import inf

# Algorithm name
ALG_UNIVERSAL_PARAM_NAME = "NAME"
# Maximum iterations
ALG_UNIVERSAL_PARAM_MAXITER = "MAXITER"
# Gradient norm tolerance
ALG_UNIVERSAL_PARAM_TOL = "TOL"
# Estimate of the Lipschitz smoothness constant
ALG_UNIVERSAL_PARAM_L_EST = "L_EST"
# Estimate of the strong convexity constant
ALG_UNIVERSAL_PARAM_MU_EST = "MU_EST"
# Estimate of the optimal value
ALG_UNIVERSAL_PARAM_OPTVAL = "OPTVAL"
# Time limit
ALG_UNIVERSAL_PARAM_TIMELIMIT = "TIMELIMIT"

# Algorithm-specific parameters
# Heavy ball momentum
ALG_HEAVY_BALL_MOMENTUM = "HEAVY BALL MOMENTUM"
# AdaGrad learning rate
ALG_ADAGRAD_LEARNING_RATE = "ADAGRAD LEARNING RATE"
# AdaGrad epsilon
ALG_ADAGRAD_EPSILON = "ADAGRAD EPSILON"
# Adam beta1
ALG_ADAM_BETA1 = "ADAM BETA1"
# Adam beta2
ALG_ADAM_BETA2 = "ADAM BETA2"
# Adam epsilon
ALG_ADAM_EPSILON = "ADAM EPSILON"
# Adam learning rate
ALG_ADAM_LEARNING_RATE = "ADAM LEARNING RATE"

# BFGS memory size
ALG_LBFGS_MEMORY_SIZE = "L-BFGS MEMORY SIZE"

# ADP GD parameters
ALG_ADP_GD_EPS = "EPS"
ALG_ADP_GD_LR0 = "LR0"

# ADP GDACC parameters
ALG_ADP_GDACC_ALR = "ALR"
ALG_ADP_GDACC_AMU = "AMU"
ALG_ADP_GDACC_BLR = "BLR"
ALG_ADP_GDACC_BMU = "BMU"

# Hypergradient descent 
ALG_HDM_NONCONVEX = "HDM NONCONVEX"
ALG_HDM_LEARNING_RATE = "HDM LEARNING RATE"
ALG_HDM_BETA_LEARNING_RATE = "HDM BETA LEARNING RATE"

# Candidate scaling matrix set
ALG_HDM_LR_VERSION = "HDM LR VERSION"
ALG_HDM_BETA_VERSION = "HDM BETA VERSION"
ALG_HDM_VERSION_DIAG = "HDM DIAGONAL VERSION"
ALG_HDM_VERSION_MATRIX = "HDM MATRIX VERSION"
ALG_HDM_VERSION_SCALAR = "HDM SCALAR VERSION"
ALG_HDM_LOGGING = "HDM LOGGING"

# Ablation parameters
ALG_HDM_NULLSTEP = "HDM NULLSTEP"
ALG_HDM_ONLINE_GD = "HDM ONLINE GD"

# Configuration of the optimization algorithms
gradient_descent_params = {
    ALG_UNIVERSAL_PARAM_NAME: "Gradient Descent",
    ALG_UNIVERSAL_PARAM_MAXITER: 1000,
    ALG_UNIVERSAL_PARAM_TOL: 1e-06,
    ALG_UNIVERSAL_PARAM_L_EST: inf,
    ALG_UNIVERSAL_PARAM_MU_EST: 0,
    ALG_UNIVERSAL_PARAM_OPTVAL: -inf
}

gradient_descent_ls_params = gradient_descent_params.copy()

gradient_descent_heavy_ball_params = {
    ALG_UNIVERSAL_PARAM_NAME: "Gradient Descent with Momemtum",
    ALG_UNIVERSAL_PARAM_MAXITER: 1000,
    ALG_UNIVERSAL_PARAM_TOL: 1e-06,
    ALG_UNIVERSAL_PARAM_L_EST: inf,
    ALG_UNIVERSAL_PARAM_MU_EST: 0,
    ALG_UNIVERSAL_PARAM_OPTVAL: -inf,
    ALG_HEAVY_BALL_MOMENTUM: 0.995
}

accelerated_gradient_descent_scvx_params = {
    ALG_UNIVERSAL_PARAM_NAME: "Accelerated Gradient Descent with Strong Convexity",
    ALG_UNIVERSAL_PARAM_MAXITER: 1000,
    ALG_UNIVERSAL_PARAM_TOL: 1e-06,
    ALG_UNIVERSAL_PARAM_L_EST: inf,
    ALG_UNIVERSAL_PARAM_MU_EST: 0,
    ALG_UNIVERSAL_PARAM_OPTVAL: -inf
}

accelerated_gradient_descent_cvx_params = {
    ALG_UNIVERSAL_PARAM_NAME: "Accelerated Gradient Descent with Convexity",
    ALG_UNIVERSAL_PARAM_MAXITER: 1000,
    ALG_UNIVERSAL_PARAM_TOL: 1e-06,
    ALG_UNIVERSAL_PARAM_L_EST: inf,
    ALG_UNIVERSAL_PARAM_MU_EST: 0,
    ALG_UNIVERSAL_PARAM_OPTVAL: -inf
}

accelearted_gradient_descent_cvx_ls_params = accelerated_gradient_descent_cvx_params.copy()

adagrad_params = {
    ALG_UNIVERSAL_PARAM_NAME: "AdaGrad",
    ALG_UNIVERSAL_PARAM_MAXITER: 1000,
    ALG_UNIVERSAL_PARAM_TOL: 1e-06,
    ALG_UNIVERSAL_PARAM_L_EST: inf,
    ALG_UNIVERSAL_PARAM_MU_EST: 0,
    ALG_UNIVERSAL_PARAM_OPTVAL: -inf,
    ALG_ADAGRAD_LEARNING_RATE: 0.1,
    ALG_ADAGRAD_EPSILON: 1e-08
}

adam_params = {
    ALG_UNIVERSAL_PARAM_NAME: "Adam",
    ALG_UNIVERSAL_PARAM_MAXITER: 1000,
    ALG_UNIVERSAL_PARAM_TOL: 1e-06,
    ALG_UNIVERSAL_PARAM_L_EST: inf,
    ALG_UNIVERSAL_PARAM_MU_EST: 0,
    ALG_UNIVERSAL_PARAM_OPTVAL: -inf,
    ALG_ADAM_BETA1: 0.9,
    ALG_ADAM_BETA2: 0.999,
    ALG_ADAM_EPSILON: 1e-08,
    ALG_ADAM_LEARNING_RATE: 1e-03
}

bfgs_params = {
    ALG_UNIVERSAL_PARAM_NAME: "BFGS",
    ALG_UNIVERSAL_PARAM_MAXITER: 1000,
    ALG_UNIVERSAL_PARAM_TOL: 1e-06,
    ALG_UNIVERSAL_PARAM_L_EST: inf,
    ALG_UNIVERSAL_PARAM_MU_EST: 0,
    ALG_UNIVERSAL_PARAM_OPTVAL: -inf,
}

lbfgs_params = {
    ALG_UNIVERSAL_PARAM_NAME: "L-BFGS",
    ALG_UNIVERSAL_PARAM_MAXITER: 1000,
    ALG_UNIVERSAL_PARAM_TOL: 1e-06,
    ALG_UNIVERSAL_PARAM_L_EST: inf,
    ALG_UNIVERSAL_PARAM_MU_EST: 0,
    ALG_UNIVERSAL_PARAM_OPTVAL: -inf,
    ALG_LBFGS_MEMORY_SIZE: 2
}

adpgd_params = {
    ALG_UNIVERSAL_PARAM_NAME: "Gradient Descent without Descent",
    ALG_UNIVERSAL_PARAM_MAXITER: 1000,
    ALG_UNIVERSAL_PARAM_TOL: 1e-06,
    ALG_UNIVERSAL_PARAM_L_EST: inf,
    ALG_UNIVERSAL_PARAM_MU_EST: 0,
    ALG_UNIVERSAL_PARAM_OPTVAL: -inf,
    ALG_ADP_GD_EPS: 1e-10,
    ALG_ADP_GD_LR0: 1e-03
}

adpgd_acc_params = {
    ALG_UNIVERSAL_PARAM_NAME: "Accelerated Gradient Descent without Descent",
    ALG_UNIVERSAL_PARAM_MAXITER: 1000,
    ALG_UNIVERSAL_PARAM_TOL: 1e-06,
    ALG_UNIVERSAL_PARAM_L_EST: inf,
    ALG_UNIVERSAL_PARAM_MU_EST: 0,
    ALG_UNIVERSAL_PARAM_OPTVAL: -inf,
    ALG_ADP_GDACC_ALR: 0.5,
    ALG_ADP_GDACC_AMU: 0.5,
    ALG_ADP_GDACC_BLR: 0.5,
    ALG_ADP_GDACC_BMU: 0.5
    
}

hdm_params = {
    ALG_UNIVERSAL_PARAM_NAME: "Hypergradient descent method",
    ALG_UNIVERSAL_PARAM_MAXITER: 1000,
    ALG_UNIVERSAL_PARAM_TOL: 1e-06,
    ALG_UNIVERSAL_PARAM_L_EST: inf,
    ALG_UNIVERSAL_PARAM_MU_EST: 0,
    ALG_UNIVERSAL_PARAM_OPTVAL: -inf,
    ALG_HDM_LEARNING_RATE: -1.0,
    ALG_HDM_BETA_LEARNING_RATE: 10.0,
    ALG_HDM_LR_VERSION: ALG_HDM_VERSION_DIAG,
    ALG_HDM_BETA_VERSION: ALG_HDM_VERSION_DIAG,
    ALG_HDM_LOGGING: False,
    ALG_HDM_NONCONVEX: False,
    ALG_HDM_NULLSTEP: True, # Whether to use null step. Always set to True unless in ablation
    ALG_HDM_ONLINE_GD: False  # Whether to use online gradient descent. Always set to False unless in ablation
}

if __name__ == "__main__":
    
    alg_params = [gradient_descent_params, gradient_descent_heavy_ball_params, accelerated_gradient_descent_scvx_params,
                  accelerated_gradient_descent_cvx_params, adagrad_params, adam_params, bfgs_params, lbfgs_params, hdm_params]
    
    for alg in alg_params:
        print(alg[ALG_UNIVERSAL_PARAM_NAME])
        for key, value in alg.items():
            if key != ALG_UNIVERSAL_PARAM_NAME:
                print(f"{key}: {value}")
        print("\n")