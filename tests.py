import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from constants import PARAM_LABEL_MAP, PARAM_ORDER, PARAM_NAMES

def get_AIC(chi2, n_params):
    """
    Calculates the Akaike Information Criterion (AIC) value.

    Parameters
    ----------
    chi2 : float
        Total chi squared value.
    n_params : int
        Number of parameters.

    Returns
    -------
    float
        AIC value.
    """
    aic = chi2 + 2 * n_params
    return aic

def get_BIC(chi2, n_params, n_obs):
    """
    Calculates the Bayesian Information Criterion (BIC) value.

    Parameters
    ----------
    chi2 : float
        Total chi squared value.
    n_params : int
        Number of parameters.
    n_obs : int
        Number of observations.

    Returns
    -------
    float
        BIC value.
    """
    bic = chi2 + n_params * np.log(n_obs)
    return bic

def plot_convergence_diagnostics(chain_dict, params_to_plot, runs_to_plot, model):
    """
    Planck/DES-style convergence diagnostics:
    - Left: trace plots (subset of walkers)
    - Right: running mean (stability)
    """

    n_params = len(params_to_plot)
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 3 * n_params))

    if n_params == 1:
        axes = axes[None, :]

    for i, param in enumerate(params_to_plot):
        idx = PARAM_ORDER.index(param)

        ax_trace = axes[i, 0]
        ax_mean  = axes[i, 1]

        for run_name, info in runs_to_plot.items():
            chain = chain_dict[run_name]   # (n_walkers, n_steps, n_params)

            n_walkers, n_steps, _ = chain.shape

            # --- TRACE PLOT (subset of walkers)
            for w in range(min(8, n_walkers)):
                ax_trace.plot(
                    chain[w, :, idx],
                    color=info["colour"],
                    alpha=0.3,
                    lw=0.8
                )

            # --- RUNNING MEAN (flattened over walkers)
            samples = chain[:, :, idx].reshape(-1)

            steps = np.arange(1, len(samples) + 1)
            running_mean = np.cumsum(samples) / steps

            ax_mean.plot(
                steps,
                running_mean,
                color=info["colour"],
                label=info["label"]
            )

        # Labels
        ax_trace.set_ylabel(PARAM_LABEL_MAP[param])
        ax_trace.set_title(f"{PARAM_LABEL_MAP[param]} Trace")
        ax_trace.grid(True)

        ax_mean.set_title(f"{PARAM_LABEL_MAP[param]} Running Mean")
        ax_mean.grid(True)

    axes[-1, 0].set_xlabel("Step")
    axes[-1, 1].set_xlabel("Samples")

    # Legend only once
    handles = [
        plt.Line2D([0], [0], color=info["colour"], label=info["label"])
        for info in runs_to_plot.values()
    ]
    axes[0, 1].legend(handles=handles)

    plt.tight_layout()
    plt.savefig(f'figures/convergence_{model}.pdf')
    plt.show()