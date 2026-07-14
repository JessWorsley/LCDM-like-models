import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import inspect

model_1 = pd.read_csv(f'results/mcmc_parameter_summary_model_1_20000steps.csv')
model_2 = pd.read_csv(f'results/mcmc_parameter_summary_model_2_20000steps.csv')
model_3 = pd.read_csv(f'results/mcmc_parameter_summary_model_3_50000steps.csv')

aic_cpl = 3461.335
bic_cpl = 3485.946

# =========================================================
# Find dataset combo with reduced chi2 closest to 1
# =========================================================

def sort_chi2(model):
    model['chi2_red_diff'] = np.abs(1 - model['chi2_red'])
    print(model.sort_values(by='chi2_red_diff', ignore_index=True)[['dataset', 'chi2_red']])
    best_chi2 = model[model['chi2_red_diff'] == min(model['chi2_red_diff'])]
    print(f'The dataset combination with a reduced chi squared closest to unity is {best_chi2['dataset'].to_string(index=False)}.\n')

print('Model 1')
sort_chi2(model_1)

print('Model 2')
sort_chi2(model_2)

print('Model 3')
sort_chi2(model_3)

# =========================================================
# Compare information criteria (of data with best chi2)
# =========================================================

def compare_two_aic(aic_a, aic_b):

    aic_min = min(aic_a, aic_b)

    delta_a = aic_a - aic_min
    delta_b = aic_b - aic_min

    if delta_a == 0:
        print('Model A is favoured...')
        if delta_b < 2:
            print('negligibly.')
        elif 2 <= delta_b < 6:
            print('weakly/moderately.')
        elif 6 <= delta_b < 10:
            print('strongly.')
        else:
            print('very strongly.')

    if delta_b == 0:
        print('Model B is favoured...')
        if delta_a < 2:
            print('negligibly.')
        elif 2 <= delta_a < 6:
            print('weakly/moderately.')
        elif 6 <= delta_a < 10:
            print('strongly.')
        else:
            print('very strongly.')

    weight_a = np.exp(-0.5 * delta_a)
    weight_b = np.exp(-0.5 * delta_b)

    norm = weight_a + weight_b

    weight_a /= norm
    weight_b /= norm

    print(f'Weighting of A:B is {weight_a:.3f}:{weight_b:.3f}\n')

    return weight_a, weight_b

def get_akaike_weights(aic_dict):
    """
    Parameters
    ----------
    aic_dict : dict
    """

    aic_values = np.array(list(aic_dict.values()))
    model_names = list(aic_dict.keys())

    aic_min = np.min(aic_values)

    deltas = aic_values - aic_min

    weights = np.exp(-0.5 * deltas)
    weights /= np.sum(weights)

    print("\nAIC comparison:")
    for name, delta, weight in zip(model_names, deltas, weights):

        if delta < 2:
            support = "substantial support"
        elif delta < 6:
            support = "moderately disfavoured"
        elif delta < 10:
            support = "strongly disfavoured"
        else:
            support = "very strongly disfavoured"

        print(
            f"{name}: "
            f"AIC = {aic_dict[name]:.3f}, "
            f"ΔAIC = {delta:.3f}, "
            f"weight = {weight:.3f} "
            f"({support})"
        )

    return dict(zip(model_names, weights))

all_data = 'Planck + Union3 + Pantheon+ + DESY5'
sn_data = 'Union3 + Pantheon+ + DESY5'
cmb_data = 'Planck'
bao_data = 'DESI'

aics = {
    "model_1": model_1[model_1['dataset'] == all_data]['AIC'].iloc[0],
    "model_2": model_2[model_2['dataset'] == all_data]['AIC'].iloc[0],
    "model_3": model_3[model_3['dataset'] == all_data]['AIC'].iloc[0],
    "cpl": aic_cpl
}

weights = get_akaike_weights(aics)