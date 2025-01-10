# Benchmark different algorithms on LIBSVM datasets
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))

import numpy as np

from algorithms.bench_algo import *
from algorithms.adagrad import AdaGrad
from algorithms.adam import Adam
from algorithms.agdcvx import AcceleratedGradientCvx
from algorithms.agdscvx import AcceleratedGradientStrongCvx
from algorithms.bfgs import SciPyBFGS
from algorithms.lbfgs import SciPyLBFGS
from algorithms.gd import GradientDescent
from algorithms.gdheavyball import GradientDescentHeavyBall
from algorithms.hdm import HyperGradientDescent

from algorithms.algo_config import *

# Load the dataset
from problems.def_problems import read_smoothed_svm_problem_from_libsvm

from utils import plot_descent_curves

import argparse
parser = argparse.ArgumentParser(description='Benchmark algorithms on different problems and datasets')
parser.add_argument('--dataset', type=str, default='./problems/a7a', help='Path to the dataset')
parser.add_argument('--plot_curves', type=int, default=1, help='Whether to plot figures')

if __name__ == "__main__":

    args = parser.parse_args()
    dataset = args.dataset
    plot_curves = args.plot_curves
    
    # np.random.seed(0)
    reg = 0.0
    L_est, n, fval, grad = read_smoothed_svm_problem_from_libsvm(file_path=dataset, reg=reg)
    
    # Get optimal value benchmark
    # bench_bfgs_params = lbfgs_params.copy()
    # bench_bfgs_params[ALG_UNIVERSAL_PARAM_TOL] = 1e-08
    # bench_bfgs_params[ALG_UNIVERSAL_PARAM_MAXITER] = 1000
    # bench_optimizer = SciPyLBFGS(bench_bfgs_params)
    
    # L-BFGS with different memory sizes
    lbfgs_params_m1 = lbfgs_params.copy()
    lbfgs_params_m1[ALG_LBFGS_MEMORY_SIZE] = 1
    lbfgs_params_m2 = lbfgs_params.copy()
    lbfgs_params_m2[ALG_LBFGS_MEMORY_SIZE] = 2
    lbfgs_params_m3 = lbfgs_params.copy()
    lbfgs_params_m3[ALG_LBFGS_MEMORY_SIZE] = 3
    lbfgs_params_m4 = lbfgs_params.copy()
    lbfgs_params_m4[ALG_LBFGS_MEMORY_SIZE] = 4
    lbfgs_params_m5 = lbfgs_params.copy()
    lbfgs_params_m5[ALG_LBFGS_MEMORY_SIZE] = 5
    lbfgs_params_m10 = lbfgs_params.copy()
    lbfgs_params_m10[ALG_LBFGS_MEMORY_SIZE] = 10
    
    hdm_params[ALG_HDM_VERSION] = ALG_HDM_VERSION_DIAG
    
    # HDM with different beta learning rates
    hdm_params_beta_lr_100 = hdm_params.copy()
    hdm_params_beta_lr_100[ALG_HDM_BETA_LEARNING_RATE] = 100.0
    hdm_params_beta_lr_10 = hdm_params.copy()
    hdm_params_beta_lr_10[ALG_HDM_BETA_LEARNING_RATE] = 10.0
    hdm_params_beta_lr_5 = hdm_params.copy()
    hdm_params_beta_lr_5[ALG_HDM_BETA_LEARNING_RATE] = 5.0
    hdm_params_beta_lr_3 = hdm_params.copy()
    hdm_params_beta_lr_3[ALG_HDM_BETA_LEARNING_RATE] = 3.0
    hdm_params_beta_lr_1 = hdm_params.copy()
    hdm_params_beta_lr_1[ALG_HDM_BETA_LEARNING_RATE] = 1.0
    
    alg_list = {
        # "AdaGrad": [AdaGrad, adagrad_params],
        "Adam": [Adam, adam_params],
        # "BFGS": [SciPyBFGS, bfgs_params],
        # "L-BFGS-M2": [SciPyLBFGS, lbfgs_params_m2],
        # "L-BFGS-M3": [SciPyLBFGS, lbfgs_params_m3],
        # "L-BFGS-M4": [SciPyLBFGS, lbfgs_params_m4],
        # "L-BFGS-M5": [SciPyLBFGS, lbfgs_params_m5],
        "L-BFGS-M10": [SciPyLBFGS, lbfgs_params_m10],
        # "GD": [GradientDescent, gradient_descent_params],
        # "GD-Polyak": [GradientDescentHeavyBall, gradient_descent_heavy_ball_params],
        # "AGD-CVX": [AcceleratedGradientCvx, accelerated_gradient_descent_cvx_params],
        # "AGD-SCVX": [AcceleratedGradientStrongCvx, accelerated_gradient_descent_cvx_params],
        # "HDM": [HyperGradientDescent, hdm_params],
        # "HDM-BetaLR100": [HyperGradientDescent, hdm_params_beta_lr_100],
        "HDM-BetaLR10": [HyperGradientDescent, hdm_params_beta_lr_10],
        "HDM-BetaLR5": [HyperGradientDescent, hdm_params_beta_lr_5],
        "HDM-BetaLR3": [HyperGradientDescent, hdm_params_beta_lr_3],
        "HDM-BetaLR1": [HyperGradientDescent, hdm_params_beta_lr_1]
    }
    
    mu_est = reg
    
    # Set algorithm parameters and initialize
    for algo in alg_list.keys():
        alg_list[algo][1][ALG_UNIVERSAL_PARAM_L_EST] = L_est
        alg_list[algo][1][ALG_UNIVERSAL_PARAM_TOL] = 1e-04
        alg_list[algo][1][ALG_UNIVERSAL_PARAM_MU_EST] = mu_est
        alg_list[algo][1][ALG_UNIVERSAL_PARAM_MAXITER] = 1500
        algo_param = alg_list[algo][1]
        algo_class = alg_list[algo][0] # type: Optimizer
        alg_list[algo].append(algo_class(params=algo_param))
    
    x0 = np.random.randn(n)
    x0 = x0 / np.linalg.norm(x0)
    
    # stats_bench = bench_optimizer.optimize(x=x0, f=fval, grad_f=grad)
    # opt_val = stats_bench[ALG_STATS_OPTIMAL_VALUE]

    # Print solver statistics
    print("%20s  %10s  %10s" % ("Solver", "nFvalCall", "nGradCall"))
    # Run the algorithms
    for algo in alg_list.keys():
        optimizer = alg_list[algo][2]
        stats = optimizer.optimize(x=x0, f=fval, grad_f=grad)
        alg_list[algo].append(stats)
        print("%20s  %10d  %10d" % (algo, stats[ALG_STATS_FEVALS], stats[ALG_STATS_GEVALS]))
        
    # Plot the descent curves
    if plot_curves:
        # alg_descent_curves = {algo: alg_list[algo][3][ALG_STATS_FUNCVALS] - opt_val for algo in alg_list.keys()}
        alg_descent_curves = {algo: alg_list[algo][3][ALG_STATS_GNORMS] for algo in alg_list.keys()}
        plot_descent_curves(alg_descent_curves, use_log_scale=True)
