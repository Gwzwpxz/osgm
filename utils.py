# Plotting utilities for visualizing the descent curves
import matplotlib.pyplot as plt
import seaborn as sns

def plot_descent_curves(algorithm_curves, 
                        xlabel="Iteration",
                        ylabel="Function Value",
                        title="Descent Curves",
                        use_log_scale=False,
                        figsize=(8, 6),
                        style="whitegrid",
                        legend_loc="best"):
    """
    Plots descent curves for multiple algorithms on a single plot.

    Parameters
    ----------
    algorithm_curves : dict
        A dictionary where keys are algorithm names (str) and values 
        are lists (or arrays) of function values across iterations.
    
    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    title : str, optional
        Title for the plot.

    use_log_scale : bool, optional
        If True, use a logarithmic scale for the y-axis.

    figsize : tuple, optional
        Figure size in inches, e.g., (width, height).

    style : str, optional
        Seaborn style to use. Options include: "white", "whitegrid", 
        "dark", "darkgrid", "ticks".

    legend_loc : str, optional
        Location of the legend, e.g., "best", "upper right", etc.

    Example
    -------
    >>> curves = {
    ...     "Gradient Descent": [10, 8, 6, 4, 2, 1, 0.5],
    ...     "Adam": [10, 7, 4, 2, 1.5, 1.0, 0.7],
    ...     "RMSProp": [10, 9, 7, 5, 3, 2, 1]
    ... }
    >>> plot_descent_curves(curves, use_log_scale=True)
    """
    sns.set_style(style)
    plt.figure(figsize=figsize)

    for algo_name, values in algorithm_curves.items():
        plt.plot(values, label=algo_name, linewidth=2)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)

    if use_log_scale:
        plt.yscale("log")
        
    # Set lower range
    # plt.ylim(bottom=1e-12)

    plt.legend(loc=legend_loc, fontsize=12)
    plt.tight_layout()
    plt.show()
