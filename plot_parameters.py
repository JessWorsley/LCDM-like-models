# =========================================================
# Imports
# =========================================================

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import pandas as pd
from cycler import cycler

from load_cc_data import load_cc
from load_bao_data import load_desi
from load_sn_data import load_union, load_pantheon, load_desy
from load_planck_data import load_planck

from distance_definitions import comoving_distance_from_H, luminosity_distance_from_H, D_V_from_DA_H, D_A_from_DM, D_H_from_H, D_M_from_chi
from analytical_models import background_analytical

from tests import get_AIC, get_BIC

from constants import SOUND_HORIZON_DRAG_EPOCH_MPC, SPEED_OF_LIGHT_KM_S, RECOMBINATION_REDSHIFT, SOUND_HORIZON_RECOMBINATION_EPOCH_MPC, LCDM, OMEGA_M0
from constants import DATASETS, PARAM_LABEL_MAP, PARAM_NAMES, PARAM_ORDER, HZ_PLOT_RUNS
from constants import NWALKERS, NSTEPS, BURN, THIN, SEED
from constants import MODEL1, MODEL2, MODEL3
from constants import IBM_PALETTE, PLOT_TEXT_OPTIONS, WONG_PALETTE

# Cycle custom palette into matplotlib
plt.rcParams['axes.prop_cycle'] = cycler(color=WONG_PALETTE[1:])
# Change text font and size for plots
plt.rcParams.update(PLOT_TEXT_OPTIONS)

# Set seed
np.random.seed(SEED)

models = {
    'model_1': pd.read_csv('results/mcmc_parameter_summary_model_1_20000steps.csv'),
    'model_2': pd.read_csv('results/mcmc_parameter_summary_model_2_20000steps.csv'),
    'model_3': pd.read_csv('results/mcmc_parameter_summary_model_3_50000steps.csv'),
}

def get_parameters(dataset):
    eps = []
    c1 = []
    c2 = []

    for model in ['model_1', 'model_2', 'model_3']:
        row = models[model].loc[models[model]['dataset'] == dataset].iloc[0]

        eps.append(row['eps_mean'])
        c1.append(row['c1_mean'])
        c2.append(row['c2_mean'])

    return eps, c1, c2

datasets = {
    'all_data': 'Planck + Union3 + Pantheon+ + DESY5',
    'desi_sn': 'Union3 + Pantheon+ + DESY5',
}

eps, c1, c2 = get_parameters(datasets['all_data'])
# eps, c1, c2 = get_parameters(datasets['desi_sn'])

all_data = 'Planck + Union3 + Pantheon+ + DESY5'
desi_sn_data = 'Union3 + Pantheon+ + DESY5'

w0_1 = models['model_1'][['wDE0_mean', 'wDE0_plus', 'wDE0_minus']].iloc[-1].tolist()
w0_2 = models['model_2'][['wDE0_mean', 'wDE0_plus', 'wDE0_minus']].iloc[-1].tolist()
w0_3 = models['model_3'][['wDE0_mean', 'wDE0_plus', 'wDE0_minus']].iloc[-1].tolist()

wa_1 = models['model_1'][['wDE0p_mean', 'wDE0p_plus', 'wDE0p_minus']].iloc[-1].tolist()
wa_2 = models['model_2'][['wDE0p_mean', 'wDE0p_plus', 'wDE0p_minus']].iloc[-1].tolist()
wa_3 = models['model_3'][['wDE0p_mean', 'wDE0p_plus', 'wDE0p_minus']].iloc[-1].tolist()

w0_cpl = [-1.023, 0.385, 0.037]
wa_cpl = [0.316, 0.032, 0.008]

w0_cpl_desi = [-0.838, 0.385, 0.037]
wa_cpl_desi = [-0.62, 0.032, 0.008]


def w(z, model, label, linestyle='-', w0=None, wa=None, eps=None, c1=None, c2=None):

    z = np.asarray(z)

    if model == 'model_1':
        w_vals = (
            2 * eps * (1 + z)**(3*eps + 3) - c1
        ) / (
            c1 + (1 + z)**3 * (2 * (1 + z)**(3*eps) - c2 )
        )
    
    elif model == 'model_2':
        sqrt = np.sqrt(8*eps + 9)

        w_vals = (
            c1 * (sqrt * (z + 1)**sqrt - 3 * (z + 1)**sqrt + sqrt + 3) - sqrt - 3
        ) / (
            6 * c1 * ((z + 1)**sqrt - 1) - 6 * c2 * (z + 1)**(1/2 * (sqrt + 3)) + 6
        )
    
    elif model == 'model_3':
        w_vals = - (
            (1 - c1) * (1 - eps)
        ) / (
            (1 - c1) + (c1 - c2) * (1 + z)**(3 - 3*eps)
        )

    elif model == 'cpl':
        w_vals = w0 + wa * z/(1 + z)

    else:
        raise ValueError(f"Unknown model: {model}")

    plt.plot(z, w_vals, label=label, linestyle=linestyle)
    return w_vals

z_eval = np.linspace(0, 2, 1000)

# w(z_eval, model='cpl', w0=w0_cpl_desi[0], wa=wa_cpl_desi[0], label='CPL (DESI)')

w(z_eval, model='model_1', eps=eps[0], c1=c1[0], c2=c2[0], label='Model I')
w(z_eval, model='model_2', eps=eps[1], c1=c1[1], c2=c2[1], label='Model II')
w(z_eval, model='model_3', eps=eps[2], c1=c1[2], c2=c2[2], label='Model III')

w(z_eval, model='cpl', w0=w0_cpl[0], wa=wa_cpl[0], label=r'$w_0w_a$CDM', linestyle='--')

plt.hlines(-1.0, np.min(z_eval), np.max(z_eval), color='k', linestyles=':', label=r'$\Lambda$CDM')
plt.xlabel(r'$z$')
plt.ylabel(r'$w_{\mathrm{DE}}(z)$')
plt.legend()
plt.xlim(z_eval[0], z_eval[-1])
plt.tight_layout()
plt.savefig('figures/evolution_wz_planck+union+pantheon+desy.pdf')
# plt.show()
plt.close()

def get_hqj(z, eps, c1, c2, model, get):

    if model == 'model_1':
        h = np.sqrt( (c1 + 2*(1 + z)**(3*eps + 3)) / (c1 + 2) )
        q = ( (3*eps + 1) * (z + 1)**(3*eps + 3) - c1 ) / ( c1 + 2*(z + 1)**(3*eps + 3) )
        j = 1 + 3*eps*(q + 1)
    elif model == 'model_2':
        sqrt = np.sqrt(9 + 8*eps)
        h = np.sqrt( c1 * (1 + z)**((3 + sqrt)/2) + (1 - c1) * (1 + z)**((3 - sqrt)/2) )
        q = ( - 1 - sqrt + c1*(1 - (1 + z)**sqrt + sqrt + (1 + z)**sqrt * sqrt) ) / ( 4 + 4*c1 * (-1 + (1 + z)**sqrt) )
        j = np.full_like(np.asarray(z, dtype=float), 1 + eps)
    elif model == 'model_3':
        h = np.sqrt( c1*(1 + z)**3 + (1 - c1)*(1 + z)**(3*eps) )
        q = 0.5 * ( c1*(1 + z)**3 - (1 - c1)*(2 - 3*eps)*(1 + z)**(3*eps) ) / ( c1*(1 + z)**3 + (1 - c1)*(1 + z)**(3*eps) )
        j = 1 + 3*eps*(q - 1/2)
    elif model == 'lcdm':
        omega_m = c1
        h = np.sqrt(omega_m*(1 + z)**3 + (1 - omega_m))
        q = (0.5 * omega_m*(1 + z)**3 - (1 - omega_m)) / (omega_m*(1 + z)**3 + (1 - omega_m))
        j = np.ones_like(np.asarray(z))

    if get == 'h':
        return h
    elif get == 'q':
        return q
    elif get == 'j':
        return j

def plot_hqj(z_eval, dataset, plot, file_name):
    eps, c1, c2 = get_parameters(dataset)

    p1 = get_hqj(z_eval, eps[0], c1[0], c2[0], 'model_1', plot)
    p2 = get_hqj(z_eval, eps[1], c1[1], c2[1], 'model_2', plot)
    p3 = get_hqj(z_eval, eps[2], c1[2], c2[2], 'model_3', plot)
    p_lcdm = get_hqj(z_eval, None, OMEGA_M0, None, 'lcdm', plot)

    p1 = get_hqj(z_eval, eps[0], c1[0], c2[0], 'model_1', f'{plot}')
    p2 = get_hqj(z_eval, eps[1], c1[1], c2[1], 'model_2', f'{plot}')
    p3 = get_hqj(z_eval, eps[2], c1[2], c2[2], 'model_3', f'{plot}')
    p_lcdm = get_hqj(z_eval, None, OMEGA_M0, None, 'lcdm', f'{plot}')

    diff1 = p1 - p_lcdm #100 * (p1 - p_lcdm) / p_lcdm
    diff2 = p2 - p_lcdm #100 * (p2 - p_lcdm) / p_lcdm
    diff3 = p3 - p_lcdm #100 * (p3 - p_lcdm) / p_lcdm

    fig = plt.figure(figsize=(6, 5))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Top panel
    ax1.plot(z_eval, p1, label='Model I', ls='-')
    ax1.plot(z_eval, p2, label='Model II', ls='-.')
    ax1.plot(z_eval, p3, label='Model III', ls=':')
    ax1.plot(z_eval, p_lcdm, label=r'$\Lambda$CDM', color='k', lw=0.8, ls='--')

    ax1.set_ylabel(rf'${plot}(z)$')
    ax1.legend()
    ax1.set_xlim(z_eval[0], z_eval[-1])
    ax1.tick_params(labelbottom=False)

    # Bottom panel
    ax2.plot(z_eval, diff1, label='I')
    ax2.plot(z_eval, diff2, label='II')
    ax2.plot(z_eval, diff3, label='III')
    ax2.axhline(0, color='k', lw=0.8, ls='--')

    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(rf'$\Delta {plot}$')

    plt.tight_layout()
    plt.subplots_adjust(left=0.16)
    plt.savefig(f'figures/evolution_{plot}z_{file_name}.pdf')
    plt.show()

for dataset_key, dataset_name in datasets.items():
    file_name = (
        dataset_name
        .replace(' ', '')
        .lower()
    )

    plot_hqj(z_eval, dataset_name, 'h', file_name)
    plot_hqj(z_eval, dataset_name, 'q', file_name)
    plot_hqj(z_eval, dataset_name, 'j', file_name)