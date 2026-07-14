# =========================================================
# Imports
# =========================================================

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
import time
import pandas as pd
from cycler import cycler

from load_cc_data import load_cc
from load_bao_data import load_desi
from load_sn_data import load_union, load_pantheon, load_desy
from load_planck_data import load_planck

from distance_definitions import comoving_distance_from_H, luminosity_distance_from_H, D_V_from_DA_H, D_A_from_DM, D_H_from_H, D_M_from_chi
from analytical_models import background_analytical, background_cpl

from tests import get_AIC, get_BIC

from constants import SOUND_HORIZON_DRAG_EPOCH_MPC, SPEED_OF_LIGHT_KM_S, RECOMBINATION_REDSHIFT, SOUND_HORIZON_RECOMBINATION_EPOCH_MPC, LCDM
from constants import DATASETS, PARAM_LABEL_MAP, PARAM_NAMES, PARAM_ORDER, HZ_PLOT_RUNS
from constants import NWALKERS, NSTEPS, BURN, THIN, SEED
from constants import MODEL1, MODEL2, MODEL3, CPL
from constants import IBM_PALETTE, PLOT_TEXT_OPTIONS

# Cycle custom palette into matplotlib
plt.rcParams['axes.prop_cycle'] = cycler(color=IBM_PALETTE)
# Change text font and size for plots
plt.rcParams.update(PLOT_TEXT_OPTIONS)

# Set seed
np.random.seed(SEED)

# Choose model
MODEL = CPL
if MODEL == "cpl":
    PARAM_NAMES = ["H0","Om0","w0","wa"]

# TODO: Need to import priors from priors.py
# TODO: Make data class

# =========================================================
# Priors and Initial Estimates
# =========================================================

# Priors from Rodrigues 2025 (changed 70->72 though)
H0_min, H0_max = 50.0, 72.0
Om0_min, Om0_max = 0.0, 1.0

q0_min, q0_max = -2.0, 2.0
j0_min, j0_max = -5.0, 5.0

eps_min, eps_max = -1.0, 5.0
c1_min, c1_max = 2.0, 8.0

wDE0_min, wDE0_max = -2.0, 0.0

H0_ini = 68
q0_ini = -0.5
j0_ini = 1.0
Om0_ini = 0.3


w0_min, w0_max = -3.0, 1.0
wa_min, wa_max = -5.0, 5.0

w0_ini = -1.0
wa_ini = 0.0


theta_initial = np.array([H0_ini, q0_ini, j0_ini, Om0_ini])

theta_initial_LCDM = np.array([H0_ini, Om0_ini, w0_ini, wa_ini])

# =========================================================
# Load Data
# =========================================================

means_cc, mu_cc, z_cc = load_cc()
means_desi, cov_desi, inv_cov_desi, z_desi, kinds_desi = load_desi()
means_union, cov_union, inv_cov_union, z_union = load_union()
means_pantheon, cov_pantheon, inv_cov_pantheon, z_pantheon = load_pantheon()
means_desy, cov_desy, inv_cov_desy, z_desy = load_desy()
means_planck, cov_planck, inv_cov_planck = load_planck()

# =========================================================
# Model Vectors
# =========================================================

def desi_model_vector(theta, H_model, z_eval_dense=None, return_all=False):
    """
    Parameters
    ----------
    theta : numpy array
        Parameter vector (H0, q0, j0, Om0, eps, c1, c2, wDE0).
    H_model : function
        Function(theta, z) that returns H(z) given theta.
    z_eval_dense : numpy array
        Dense z grid for integration, by default 0 to max(z).
    return_all : bool
        If True, also returns the raw H(z), DM(z), etc. arrays on dense grid.

    Returns
    -------
    numpy array
        Model vector for DESI.
    """

    # Choose dense grid for stable integrals
    if z_eval_dense is None:
        zmin = 0.0
        zmax = np.max(z_desi)
        z_eval_dense = np.linspace(zmin, zmax, 1000)

    H_dense = H_model(theta, z_eval=z_eval_dense)
    if H_dense is None:
        print('H(z) could not be calculated.')
        return None
    
    # Compute distances on dense grid
    chi_dense = comoving_distance_from_H(z_eval_dense, H_dense)
    DM_dense = D_M_from_chi(chi_dense)
    DA_dense = D_A_from_DM(DM_dense, z_eval_dense)
    DV_dense = D_V_from_DA_H(DA_dense, z_eval_dense, H_dense)
    DH_dense = D_H_from_H(H_dense)

    # Interpolate to BAO redshifts
    H_at_bao = np.interp(z_desi, z_eval_dense, H_dense)
    DM_at_bao = np.interp(z_desi, z_eval_dense, DM_dense)
    DV_at_bao = np.interp(z_desi, z_eval_dense, DV_dense)
    DH_at_bao = np.interp(z_desi, z_eval_dense, DH_dense)

    # Assemble model vector in the same order as desi data
    model_vec = np.empty_like(means_desi, dtype=float)
    for i, kind in enumerate(kinds_desi):
        if kind == 'DV_over_rs':
            model_vec[i] = DV_at_bao[i] / SOUND_HORIZON_DRAG_EPOCH_MPC
        elif kind == 'DM_over_rs':
            model_vec[i] = DM_at_bao[i] / SOUND_HORIZON_DRAG_EPOCH_MPC
        elif kind == 'DH_over_rs':
            # D_H / r_s = c / (H * r_s)
            model_vec[i] = DH_at_bao[i] / SOUND_HORIZON_DRAG_EPOCH_MPC
        else:
            raise ValueError(f'Unknown BAO kind: {kind}')
        
    if np.any(~np.isfinite(model_vec)):
        return None

    if return_all:
        return model_vec, dict(z_eval=z_eval_dense, H=H_dense, DM=DM_dense, DV=DV_dense, DH=DH_dense)
    return model_vec

def sn_model_vector(theta, z_eval, H_model):
    H_vals = H_model(theta, z_eval)

    if H_vals is None or np.any(~np.isfinite(H_vals)) or np.any(H_vals <= 0):
        return None

    DL_vals = luminosity_distance_from_H(z_eval, H_vals)

    if np.any(DL_vals <= 0) or np.any(~np.isfinite(DL_vals)):
        return None

    if H_vals is None or np.any(~np.isfinite(H_vals)):
        return None
    if np.any(DL_vals <= 0) or np.any(~np.isfinite(DL_vals)):
        return None
    
    mu = 5*np.log10(DL_vals) + 25 # Convert Mpc to pc
    H0 = theta[0]
    mu -= 5*np.log10(H0/70.0)

    return mu 

def planck_model_vector(theta, background,
                        z_star=RECOMBINATION_REDSHIFT,
                        z_max=1000, # previously 1200
                        n_int=1000): # previously 4000
    """
    Planck compressed distance prior vector: [R, lA]
    Consistent implementation.
    """

    # redshift grid
    z = np.linspace(0.0, z_max, n_int)

    # theta_bg = theta[:4]

    result = background(theta, z)
    if result is None:
        return None

    H, Om = result

    if H is None or np.any(~np.isfinite(H)) or np.any(H <= 0):
        return None

    # Comoving distance integral chi(z) = c int_0^z dz'/H(z')
    # Assumes flat 
    # invH = SPEED_OF_LIGHT_KM_S / H
    # chi = np.concatenate(([0], cumulative_trapezoid(invH, z)))
    chi = comoving_distance_from_H(z, H)

    if np.any(~np.isfinite(chi)):
        return None

    # interpolate distance to recombination
    chi_star = np.interp(z_star, z, chi)

    if not np.isfinite(chi_star) or chi_star <= 0:
        return None

    # parameters at z=0
    H0 = theta[0]
    Om0 = theta[3]
    # Om0 = Om[0]

    if not np.isfinite(Om0) or Om0 <= 0:
        return None

    # Distance priors
    R = np.sqrt(Om0) * H0 * chi_star / SPEED_OF_LIGHT_KM_S
    lA = np.pi * chi_star / SOUND_HORIZON_RECOMBINATION_EPOCH_MPC

    if not np.all(np.isfinite([R, lA])):
        return None

    return np.array([R, lA])

# =========================================================
# Log Likelihood Functions
# =========================================================

def loglike_desi(theta, H_model):
    """
    Computes log-likelihood for DESI BAO given theta and a model.
    ----------------------------------------------------------------
    theta:          model params
    H_model:        must accept (theta, z_eval=...) and return H(z_eval) array
    ----------------------------------------------------------------
    Returns:    log-likelihood of H_model for Gaussian uncertainties
    """
    model_vec = desi_model_vector(theta, H_model)
    model_vec = desi_model_vector(theta, H_model)

    if model_vec is None:
        # print("failure: model_vec is None")
        return -np.inf

    if np.any(~np.isfinite(model_vec)):
        # print("failure: model_vec is inf")
        return -np.inf

    # Make sure shapes are equal
    if model_vec.shape != means_desi.shape:
        raise ValueError(f'model_vec.shape {model_vec.shape} != means_desi.shape {means_desi.shape}')

    # Check how different our modelled BAO values are from the data
    resid = means_desi - model_vec
    
    # X^2 = r^T . C^-1 . r
    chi_squared = float(resid.T @ inv_cov_desi @ resid)
    if not np.isfinite(chi_squared):
        return -np.inf
    return -0.5 * chi_squared

def loglike_sn(theta, z_sn, means_sn, inv_cov_sn, H_model):
    # Make sure this returns H(z) in km/s/Mpc
    mu_model = sn_model_vector(theta, z_sn, H_model)
    if mu_model is None or np.any(~np.isfinite(mu_model)):
        print("failure: mu_model is None")
        return -np.inf

    H_vals = H_model(theta, z_sn)
    
    if H_vals is None or np.any(~np.isfinite(H_vals)) or np.any(H_vals <= 0):
        return -np.inf
       
    # Check how different our modelled values are from the data
    resid = mu_model - means_sn

    # X^2 = r^T . C^-1 . r
    chi_squared = float(resid.T @ inv_cov_sn @ resid)
    if not np.isfinite(chi_squared):
        return -np.inf
    
    return -0.5 * chi_squared

def loglike_planck(theta, background, planck_mean, planck_inv_cov):

    v_th = planck_model_vector(theta, background)

    if v_th is None:
        # print("Planck fail at", theta)
        return -np.inf

    if v_th.shape != planck_mean.shape:
        return -np.inf

    if np.any(~np.isfinite(v_th)):
        print("failure: v_th is inf")
        return -np.inf

    delta = v_th - planck_mean
    chi2 = delta @ planck_inv_cov @ delta

    if not np.isfinite(chi2):
        return -np.inf

    return -0.5 * chi2

# =========================================================
# Priors
# =========================================================

def logprior(theta, theta_min, theta_max, model=MODEL):
    if model == 'cpl':
        H0, Om0, w0, wa = theta
        H0_min, Om0_min, w0_min, wa_min = theta_min
        H0_max, Om0_max, w0_max, wa_max = theta_max

        if not (H0_min < H0 < H0_max):
            return -np.inf

        if not (Om0_min < Om0 < Om0_max):
            return -np.inf

        if not (w0_min < w0 < w0_max):
            return -np.inf

        if not (wa_min < wa < wa_max):
            return -np.inf

        return 0.0

    H0, q0, j0, Om0 = theta
    H0_min, q0_min, j0_min, Om0_min = theta_min
    H0_max, q0_max, j0_max, Om0_max = theta_max

    if model == 'model_2':
        eps = j0 - 1
        # Avoid negatives occurring in square root
        if 9 + 8*eps <= 0:
            return -np.inf
        
    if model == 'model_3':
      
        # Avoid denominator issues
        if abs(2*j0 - 6*q0 + 1) < 1e-2:
            return -np.inf

        # Avoid exponent blow-up
        if abs(2*q0 - 1) < 1e-2:
            return -np.inf

        # Ensure numerator positive at high z
        if j0 < q0*(2*q0 + 1):
            return -np.inf

        # Keep exponent reasonable
        exponent = 2*(j0 - 1)/(2*q0 - 1)
        if abs(exponent) > 10:
            return -np.inf

        # Enforce matter domination at high z
        if j0 < 0.5:
            return -np.inf
        
    if not (H0_min < H0 < H0_max):
        return -np.inf
    if not (q0_min < q0 < q0_max):
        return -np.inf
    if not (j0_min < j0 < j0_max):
        return -np.inf
    if not (Om0_min < Om0 < Om0_max):
        return -np.inf
    
    return 0.0

# =========================================================
# Joint Log Posterior
# =========================================================

def H_only(background):

    def H_model(*args, **kwargs):
        result = background(*args, **kwargs)

        if result is None:
            return None

        if not isinstance(result, tuple):
            print("ERROR: Expected tuple, got", type(result))
            return None

        return result[0]

    return H_model

def background_model(theta, z_eval):
    if MODEL == "cpl":
        return background_cpl(theta, z_eval)
    
    result = background_analytical(theta, z_eval, model=MODEL)

    if result is None:
        return None

    if not isinstance(result, tuple) or len(result) != 2:
        return None

    H, Om = result

    if np.any(~np.isfinite(H)) or np.any(H <= 0):
        return None
    
    return H, Om

def logposterior(theta, data_combo=None, model=MODEL):
    """
    Returns the joint logarithmic posterior for DESI and SN datasets by adding the log prior and log likelihood.

    Arguments
    ----------
    theta :         numpy array
                    Parameter vector.
    data_combo :    string | None
                    SN dataset to be combined with DESI (None / 'union' / 'pantheon' / 'desy').

    Returns
    ----------
    float
        Logarithmic posterior.
    """

    if MODEL == "cpl":
        lp = logprior(theta, theta_min=(H0_min,Om0_min,w0_min,wa_min), theta_max=(H0_max,Om0_max,w0_max,wa_max), model=model)
    else:
        lp = logprior(theta, theta_min=(H0_min,q0_min,j0_min,Om0_min), theta_max=(H0_max,q0_max,j0_max,Om0_max), model=model)
    if not np.isfinite(lp):
        return -np.inf
    
    # DESI likelihood
    ll_desi = loglike_desi(theta, H_only(background_model))
    # If likelihood is invalid, reject this sample
    if not np.isfinite(ll_desi):
        return -np.inf

    ll = ll_desi

    # SN likelihoods
    if data_combo and 'union' in data_combo:
        ll_union = loglike_sn(theta, z_union, means_union, inv_cov_union, H_only(background_model))
        if not np.isfinite(ll_union):
            return -np.inf
        ll += ll_union

    if data_combo and 'pantheon' in data_combo:
        ll_pantheon = loglike_sn(theta, z_pantheon, means_pantheon, inv_cov_pantheon, H_only(background_model))
        if not np.isfinite(ll_pantheon):
            return -np.inf
        ll += ll_pantheon

    if data_combo and 'desy' in data_combo:
        ll_desy = loglike_sn(theta, z_desy, means_desy, inv_cov_desy, H_only(background_model))
        if not np.isfinite(ll_desy):
            return -np.inf
        ll += ll_desy
    
    # Planck likelihood
    if data_combo and 'planck' in data_combo:
        ll_planck = loglike_planck(theta, background_model, means_planck, inv_cov_planck)
        if not np.isfinite(ll_planck):
            return -np.inf
        ll += ll_planck
   
    if not np.isfinite(lp + ll):
        print("BAD theta:", theta)

    return lp + ll

    
# =========================================================
# Emcee Setup, Run & Results
# =========================================================

def print_convergence_stats(chain):
    tau = emcee.autocorr.integrated_time(chain, quiet=True)
    n_walkers, n_steps, _ = chain.shape

    print("\nConvergence diagnostics:")
    for i, name in enumerate(PARAM_NAMES):
        Neff = (n_walkers * n_steps) / tau[i]
        print(f"{name:6s}: tau = {tau[i]:.1f},  N/tau = {(n_steps / tau[i]):.1f},  Neff = {Neff:.0f}")

def is_physical(theta):
    z_test = np.linspace(0, 2.5, 50)
    result = background_model(theta, z_test)

    if result is None:
        return False

    H, _ = result

    if np.any(~np.isfinite(H)) or np.any(H <= 0):
        return False

    return True

def run_mcmc(initial_values, data_combo=None, model=MODEL, central_value='median'):
    print(f'{model.upper()}:')
    print(f'DESI + {data_combo}' if data_combo else 'DESI')
    ndim = len(initial_values)
    pos = np.array(initial_values, copy=True) + 1e-3 * np.random.randn(NWALKERS, ndim) # previously 1e-3

    sampler = emcee.EnsembleSampler(NWALKERS, ndim, logposterior, args=(data_combo, model))

    t0 = time.time()
    state = sampler.run_mcmc(pos, NSTEPS, progress=True)
    t1 = time.time()
    print(f'MCMC completed in {round((t1-t0)/60, 1)} s.')

    chain = sampler.get_chain()  # (n_walkers, n_steps, ndim)
    samples = sampler.get_chain(discard=BURN, thin=THIN, flat=True)
    print('Kept samples shape:', samples.shape)
    print("Acceptance fraction:", np.mean(sampler.acceptance_fraction))

    if model == 'cpl':
        H0 = samples[:,0]
        Om0 = samples[:,1]
        w0 = samples[:,2]
        wa = samples[:,3]

        if central_value == 'median':
            means = np.median(samples, axis=0)
        elif central_value == 'mean':
            means = np.mean(samples, axis=0)
        else:
            raise ValueError("central_value only accepts arguments 'mean' or 'median'.")

        # 95% (2-sigma) central credible intervals
        low95, high95 = np.percentile(samples, [2.5, 97.5], axis=0)

        # Mean and 2-sigma limits
        print('\nParameter summary:')
        for i, name in enumerate(PARAM_NAMES):
            print(f'{name:6s} = {means[i]:.3f} (+{high95[i]-means[i]:.3f}/-{means[i]-low95[i]:.3f})  (2-sigma range: {low95[i]:.3f} – {high95[i]:.3f})')
        
        summary = {}
        # parameters in the chain
        for i, name in enumerate(PARAM_NAMES):
            summary[name] = {
                "mean": means[i],
                "plus": high95[i] - means[i],
                "minus": means[i] - low95[i]
        }
            
        # Best-fit parameters
        theta_best = means

        # Choose analytical H model for this run
        H_model = H_only(background_model)
    
        # Instead of using means/medians for chi-squared, the maximum likelihood might be a better choice?
        chi_squared_desi = -2 * loglike_desi(theta_best, H_model)

        chi_squared_planck = 0
        chi_squared_sn = 0

        N_total = len(means_desi)

        if data_combo and 'planck' in data_combo:
            chi_squared_planck = -2 * loglike_planck(
                theta_best, background_model,
                means_planck, inv_cov_planck
            )
            N_total += len(means_planck)

        # Minus 1 from SN data because of marginalising over nuisance parameter M
        if data_combo and 'union' in data_combo:
            chi_squared_sn += -2 * loglike_sn(
                theta_best, z_union, means_union,
                inv_cov_union, H_model
            )
            N_total += len(means_union) - 1

        if data_combo and 'pantheon' in data_combo:
            chi_squared_sn += -2 * loglike_sn(
                theta_best, z_pantheon, means_pantheon,
                inv_cov_pantheon, H_model
            )
            N_total += len(means_pantheon) - 1

        if data_combo and 'desy' in data_combo:
            chi_squared_sn += -2 * loglike_sn(
                theta_best, z_desy, means_desy,
                inv_cov_desy, H_model
            )
            N_total += len(means_desy) - 1

        chi_squared_total = chi_squared_desi + chi_squared_planck + chi_squared_sn

        # Degrees of freedom
        dof = N_total - len(theta_best)

        chi_squared_red = chi_squared_total / dof

        print('\nGoodness-of-fit:')
        print(f'BAO chi-squared = {chi_squared_desi:.3f}')
        print(f'SN chi-squared = {chi_squared_sn:.3f}')
        print(f'Planck chi-squared = {chi_squared_planck:.3f}')
        print(f'Total chi-squared = {chi_squared_total:.3f}')
        print(f'Reduced total chi-squared = {chi_squared_red:.3f}')

        print_convergence_stats(chain)

        aic = get_AIC(chi_squared_total, len(theta_best))
        bic = get_BIC(chi_squared_total, len(theta_best), N_total)

        print(f'AIC: {aic:.4f}')
        print(f'BIC: {bic:.4f}')

        summary["AIC"] = aic
        summary["BIC"] = bic

        samples_all = np.column_stack((H0, Om0, w0, wa))
        return sampler, chain, samples_all, summary, chi_squared_red

    H0 = samples[:,0]
    q0 = samples[:,1]
    j0 = samples[:,2]
    Om0 = samples[:,3]

    if model == 'model_1':
        eps = (j0 - 1) / (3 * (1 + q0))
        c1 = (j0 - 2 * q0**2 - q0) / (q0 + 1)**2
        c2 = Om0 * (j0 + 3*q0 + 2) / (q0 + 1)**2

    if model == 'model_2':
        eps = j0 - 1
        c1 = ( 4 * np.sqrt(8*j0 + 1) * q0 + 8*j0 + np.sqrt(8*j0 + 1) + 1)/( 2 * (8*j0 + 1) )
        c2 = Om0

    if model == 'model_3':
        eps = 2*(1-j0)/(3*(1-2*q0))
        c1 = 2*(j0 - q0 - 2*q0**2)/(1 - 6*q0 + 2*j0)
        c2 = Om0

    wDE0 = (1 - 2*q0)/(3 * (Om0 - 1))
    wDE0p = - (2*j0 * (Om0 - 1) + 2*q0*(2*q0 - 3*Om0 + 1) + Om0)/(3 * (Om0 - 1)**2)
      
    # Decide whether to use medians or means for central values (I know it all says mean don't @ me)
    if central_value == 'median':
        means = np.median(samples, axis=0)
        mean_eps, mean_c1, mean_c2, mean_wDE0, mean_wDE0p = np.median(eps), np.median(c1), np.median(c2), np.median(wDE0), np.median(wDE0p)
    elif central_value == 'mean':
        means = np.mean(samples, axis=0)
        mean_eps, mean_c1, mean_c2, mean_wDE0, mean_wDE0p = np.mean(eps), np.mean(c1), np.mean(c2), np.mean(wDE0), np.mean(wDE0p)
    else:
        raise ValueError("central_value only accepts arguments 'mean' or 'median'.")

    # 95% (2-sigma) central credible intervals
    low95, high95 = np.percentile(samples, [2.5, 97.5], axis=0)

    # Inferred quantities
    low95_eps, high95_eps = np.percentile(eps, [2.5, 97.5], axis=0)
    low95_c1, high95_c1 = np.percentile(c1, [2.5, 97.5], axis=0)
    low95_c2, high95_c2 = np.percentile(c2, [2.5, 97.5], axis=0)
    low95_wDE0, high95_wDE0 = np.percentile(wDE0, [2.5, 97.5], axis=0)
    low95_wDE0p, high95_wDE0p = np.percentile(wDE0p, [2.5, 97.5], axis=0)

    # Mean and 2-sigma limits
    print('\nParameter summary:')
    for i, name in enumerate(PARAM_NAMES):
        print(f'{name:6s} = {means[i]:.3f} (+{high95[i]-means[i]:.3f}/-{means[i]-low95[i]:.3f})  (2-sigma range: {low95[i]:.3f} – {high95[i]:.3f})')
    print(f'epsilon = {mean_eps:.3f} (+{high95_eps-mean_eps:.3f}/-{mean_eps-low95_eps:.3f})  (2-sigma range: {low95_eps:.3f} – {high95_eps:.3f})')
    print(f'c1 = {mean_c1:.3f} (+{high95_c1-mean_c1:.3f}/-{mean_c1-low95_c1:.3f})  (2-sigma range: {low95_c1:.3f} – {high95_c1:.3f})')
    print(f'c2 = {mean_c2:.3f} (+{high95_c2-mean_c2:.3f}/-{mean_c2-low95_c2:.3f})  (2-sigma range: {low95_c2:.3f} – {high95_c2:.3f})')
    print(f'wDE0 = {mean_wDE0:.3f} (+{high95_wDE0-mean_wDE0:.3f}/-{mean_wDE0-low95_wDE0:.3f})  (2-sigma range: {low95_wDE0:.3f} – {high95_wDE0:.3f})')
    print(f'wDE0p = {mean_wDE0p:.3f} (+{high95_wDE0p-mean_wDE0p:.3f}/-{mean_wDE0p-low95_wDE0p:.3f})  (2-sigma range: {low95_wDE0p:.3f} – {high95_wDE0p:.3f})')

    summary = {}
    # parameters in the chain
    for i, name in enumerate(PARAM_NAMES):
        summary[name] = {
            "mean": means[i],
            "plus": high95[i] - means[i],
            "minus": means[i] - low95[i]
    }

    # derived parameters
    summary["eps"] = {
        "mean": mean_eps,
        "plus": high95_eps - mean_eps,
        "minus": mean_eps - low95_eps
    }

    summary["c1"] = {
        "mean": mean_c1,
        "plus": high95_c1 - mean_c1,
        "minus": mean_c1 - low95_c1
    }

    summary["c2"] = {
        "mean": mean_c2,
        "plus": high95_c2 - mean_c2,
        "minus": mean_c2 - low95_c2
    }

    summary["wDE0"] = {
        "mean": mean_wDE0,
        "plus": high95_wDE0 - mean_wDE0,
        "minus": mean_wDE0 - low95_wDE0
    }

    summary["wDE0p"] = {
        "mean": mean_wDE0p,
        "plus": high95_wDE0p - mean_wDE0p,
        "minus": mean_wDE0p - low95_wDE0p
    }

    # Best-fit parameters
    theta_best = means

    # Choose analytical H model for this run
    H_model = H_only(background_model)
 
    # Instead of using means/medians for chi-squared, the maximum likelihood might be a better choice?
    chi_squared_desi = -2 * loglike_desi(theta_best, H_model)

    chi_squared_planck = 0
    chi_squared_sn = 0

    N_total = len(means_desi)

    if data_combo and 'planck' in data_combo:
        chi_squared_planck = -2 * loglike_planck(
            theta_best, background_model,
            means_planck, inv_cov_planck
        )
        N_total += len(means_planck)

    # Minus 1 from SN data because of marginalising over nuisance parameter M
    if data_combo and 'union' in data_combo:
        chi_squared_sn += -2 * loglike_sn(
            theta_best, z_union, means_union,
            inv_cov_union, H_model
        )
        N_total += len(means_union) - 1

    if data_combo and 'pantheon' in data_combo:
        chi_squared_sn += -2 * loglike_sn(
            theta_best, z_pantheon, means_pantheon,
            inv_cov_pantheon, H_model
        )
        N_total += len(means_pantheon) - 1

    if data_combo and 'desy' in data_combo:
        chi_squared_sn += -2 * loglike_sn(
            theta_best, z_desy, means_desy,
            inv_cov_desy, H_model
        )
        N_total += len(means_desy) - 1

    chi_squared_total = chi_squared_desi + chi_squared_planck + chi_squared_sn

    # Degrees of freedom
    dof = N_total - len(theta_best)

    chi_squared_red = chi_squared_total / dof

    print('\nGoodness-of-fit:')
    print(f'BAO chi-squared = {chi_squared_desi:.3f}')
    print(f'SN chi-squared = {chi_squared_sn:.3f}')
    print(f'Planck chi-squared = {chi_squared_planck:.3f}')
    print(f'Total chi-squared = {chi_squared_total:.3f}')
    print(f'Reduced total chi-squared = {chi_squared_red:.3f}')

    print_convergence_stats(chain)

    aic = get_AIC(chi_squared_total, len(theta_best))
    bic = get_BIC(chi_squared_total, len(theta_best), N_total)

    print(f'AIC: {aic:.4f}')
    print(f'BIC: {bic:.4f}')

    summary["AIC"] = aic
    summary["BIC"] = bic

    samples_all = np.column_stack((H0, q0, j0, Om0, eps, c1, c2, wDE0, wDE0p))
    return sampler, chain, samples_all, summary, chi_squared_red

samples_dict = {}
chain_dict = {}
results_summary = {}
chi_squared_dict = {}

# If theta_initial is a mutable object, some MCMC codes mutate the initial state, so the next run gets a shifted starting point. Therefore using copies.
# _, chain_dict['desi'], samples_dict['desi'], results_summary['desi'], chi_squared_dict['desi'] = run_mcmc(theta_initial_LCDM.copy(), model=MODEL)
# _, chain_dict['union'], samples_dict['union'], results_summary['union'], chi_squared_dict['union'] = run_mcmc(theta_initial_LCDM.copy(), data_combo=['union'], model=MODEL)
# _, chain_dict['pantheon'], samples_dict['pantheon'], results_summary['pantheon'], chi_squared_dict['pantheon'] = run_mcmc(theta_initial_LCDM.copy(), data_combo=['pantheon'], model=MODEL)
# _, chain_dict['desy'], samples_dict['desy'], results_summary['desy'], chi_squared_dict['desy'] = run_mcmc(theta_initial_LCDM.copy(), data_combo=['desy'], model=MODEL)
# _, chain_dict['union+pantheon+desy'], samples_dict['union+pantheon+desy'], results_summary['union+pantheon+desy'], chi_squared_dict['union+pantheon+desy'] = run_mcmc(theta_initial_LCDM.copy(), data_combo=['union','pantheon','desy'], model=MODEL)

# _, chain_dict['planck'], samples_dict['planck'], results_summary['planck'], chi_squared_dict['planck'] = run_mcmc(theta_initial_LCDM.copy(), data_combo=['planck'], model=MODEL)
# _, chain_dict['planck+union'], samples_dict['planck+union'], results_summary['planck+union'], chi_squared_dict['planck+union'] = run_mcmc(theta_initial_LCDM.copy(), data_combo=['planck','union'], model=MODEL)
# _, chain_dict['planck+pantheon'], samples_dict['planck+pantheon'], results_summary['planck+pantheon'], chi_squared_dict['planck+pantheon'] = run_mcmc(theta_initial_LCDM.copy(), data_combo=['planck','pantheon'], model=MODEL)
# _, chain_dict['planck+desy'], samples_dict['planck+desy'], results_summary['planck+desy'], chi_squared_dict['planck+desy'] = run_mcmc(theta_initial_LCDM.copy(), data_combo=['planck','desy'], model=MODEL)
# _, chain_dict['planck+union+pantheon+desy'], samples_dict['planck+union+pantheon+desy'], results_summary['planck+union+pantheon+desy'], chi_squared_dict['planck+union+pantheon+desy'] = run_mcmc(theta_initial_LCDM.copy(), data_combo=['planck','union','pantheon','desy'], model=MODEL)


# _, chain_dict['desi'], samples_dict['desi'], results_summary['desi'], chi_squared_dict['desi'] = run_mcmc(theta_initial.copy(), model=MODEL)
# _, chain_dict['union'], samples_dict['union'], results_summary['union'], chi_squared_dict['union'] = run_mcmc(theta_initial.copy(), data_combo=['union'], model=MODEL)
# _, chain_dict['pantheon'], samples_dict['pantheon'], results_summary['pantheon'], chi_squared_dict['pantheon'] = run_mcmc(theta_initial.copy(), data_combo=['pantheon'], model=MODEL)
# _, chain_dict['desy'], samples_dict['desy'], results_summary['desy'], chi_squared_dict['desy'] = run_mcmc(theta_initial.copy(), data_combo=['desy'], model=MODEL)
# _, chain_dict['union+pantheon+desy'], samples_dict['union+pantheon+desy'], results_summary['union+pantheon+desy'], chi_squared_dict['union+pantheon+desy'] = run_mcmc(theta_initial.copy(), data_combo=['union','pantheon','desy'], model=MODEL)

# _, chain_dict['planck'], samples_dict['planck'], results_summary['planck'], chi_squared_dict['planck'] = run_mcmc(theta_initial.copy(), data_combo=['planck'], model=MODEL)
# _, chain_dict['planck+union'], samples_dict['planck+union'], results_summary['planck+union'], chi_squared_dict['planck+union'] = run_mcmc(theta_initial.copy(), data_combo=['planck','union'], model=MODEL)
# _, chain_dict['planck+pantheon'], samples_dict['planck+pantheon'], results_summary['planck+pantheon'], chi_squared_dict['planck+pantheon'] = run_mcmc(theta_initial.copy(), data_combo=['planck','pantheon'], model=MODEL)
# _, chain_dict['planck+desy'], samples_dict['planck+desy'], results_summary['planck+desy'], chi_squared_dict['planck+desy'] = run_mcmc(theta_initial.copy(), data_combo=['planck','desy'], model=MODEL)
# _, chain_dict['planck+union+pantheon+desy'], samples_dict['planck+union+pantheon+desy'], results_summary['planck+union+pantheon+desy'], chi_squared_dict['planck+union+pantheon+desy'] = run_mcmc(theta_initial.copy(), data_combo=['planck','union','pantheon','desy'], model=MODEL)

# =========================================================
# Exporting results to tables
# =========================================================

# for name, samples in samples_dict.items():
#     df = pd.DataFrame(samples, columns=PARAM_ORDER)
#     df.to_csv(f"results/mcmc_parameter_results_{MODEL}_{name}_{NSTEPS}steps.csv", index=False)

# rows = []
# for dataset, params in results_summary.items():
#     # row = {"dataset": dataset.replace("+", " + ")}
#     row = {"dataset": " + ".join(DATASETS[p]["label"] for p in dataset.split("+"))}
#     for p in PARAM_ORDER:
#         if p in params:
#             row[f"{p}_mean"]  = params[p]["mean"]
#             row[f"{p}_plus"]  = params[p]["plus"]
#             row[f"{p}_minus"] = params[p]["minus"]
#         else:
#             row[f"{p}_mean"]  = None
#             row[f"{p}_plus"]  = None
#             row[f"{p}_minus"] = None
#     row["chi2_red"] = chi_squared_dict.get(dataset, None)

#     row["AIC"] = params.get("AIC", None)
#     row["BIC"] = params.get("BIC", None)

#     rows.append(row)

# df = pd.DataFrame(rows)
# df.to_csv(f"results/mcmc_parameter_summary_{MODEL}_{NSTEPS}steps.csv", index=False)

# # LaTeX table
# latex_rows = []
# for dataset, params in results_summary.items():
#     # row = {"Dataset": DATASETS[dataset]["label"]}
#     row = {"dataset": " + ".join(DATASETS[p]["label"] for p in dataset.split("+"))}
#     for p in PARAM_ORDER:
#         if p in params:
#             m = params[p]["mean"]
#             up = params[p]["plus"]
#             lo = params[p]["minus"]
#             row[p] = f"${m:.3f}^{{+{up:.3f}}}_{{-{lo:.3f}}}$"
#         else:
#             row[p] = "--"
#     row[r"$\chi^2_{\text{red}}$"] = f"{chi_squared_dict.get(dataset, np.nan):.3f}"
#     latex_rows.append(row)

# latex_df = pd.DataFrame(latex_rows)
# latex_df.to_latex(f'results/mcmc_parameter_summary_{MODEL}_{NSTEPS}steps.tex', index=False, escape=False)

# # =========================================================
# # Plotting: Corner plots
# # =========================================================

# def plot_corner(name, samples, filename=None):

#     has_planck = "planck" in name

#     if has_planck:
#         plot_params = ['H0','q0','j0','Om0','eps','c1','c2','wDE0','wDE0p']
#     else:
#         plot_params = ['H0','q0','j0','eps','c1']

#     indices = [PARAM_ORDER.index(p) for p in plot_params]
#     labels = [PARAM_LABEL_MAP[p] for p in plot_params]

#     samples_subset = samples[:, indices]
#     mask = np.all(np.isfinite(samples_subset), axis=1)
#     samples_subset = samples_subset[mask]

#     ranges = [
#         (np.percentile(samples_subset[:, i], 1),
#         np.percentile(samples_subset[:, i], 99))
#         for i in range(samples_subset.shape[1])
#     ]

#     fig = corner.corner(
#         samples_subset,
#         labels=labels,
#         range=ranges,
#         color="C0",
#         bins=40,
#         smooth=True,
#         smooth1d=True,
#         plot_datapoints=False,
#         fill_contours=True,
#         levels=(0.68, 0.95),
#     ) 

#     # Add dataset label
#     fig.suptitle(name.replace("+", " + ").upper(), fontsize=14)

#     if filename:
#         plt.savefig(filename, dpi=300, bbox_inches="tight")

#     plt.show()

# for name, samples in samples_dict.items():
#     plot_corner(name, samples, filename=f"figures/corner_{name}_{MODEL}_{NWALKERS}walkers_{NSTEPS}steps.pdf")

# =========================================================
# Plotting: Combined H(z) with data-inferred points
# =========================================================

# def compute_H_samples(samples, z_plot):

#     H_samples = np.array([
#         background_model(theta[:4], z_plot)[0] for theta in samples # extract H directly
#     ])

#     return H_samples

# plt.figure(figsize=(8,6))
# z_plot = np.linspace(0, 2.5, 200)

# H_samples_dict = {}

# def median_and_68(samples):
#     med = np.nanmedian(samples, axis=0)
#     lo  = np.nanpercentile(samples, 16, axis=0)
#     hi  = np.nanpercentile(samples, 84, axis=0)
#     return med, lo, hi

# # Model curves + 68% bands
# for name, info in HZ_PLOT_RUNS.items():

#     H_samples = compute_H_samples(samples_dict[name], z_plot)
#     # H_samples = H_samples_dict[name]
#     med, lo, hi = median_and_68(H_samples)

#     plt.plot(z_plot, med, lw=2, label=info["label"], color=info["colour"], alpha=0.8)

#     plt.fill_between(z_plot, lo, hi, color=info["colour"], alpha=0.2)

# # CC H(z) points
# plt.errorbar(z_cc, means_cc, yerr=mu_cc, fmt='k.', label='CC data')

# # DESI BAO-derived H(z) points
# mask_DH = kinds_desi == 'DH_over_rs'
# DH_over_rs = means_desi[mask_DH]
# sigma_DH_over_rs = np.sqrt(np.diag(cov_desi))[mask_DH]

# H_from_desi = SPEED_OF_LIGHT_KM_S / (DH_over_rs * SOUND_HORIZON_DRAG_EPOCH_MPC)
# H_err_from_desi = SPEED_OF_LIGHT_KM_S * sigma_DH_over_rs / (DH_over_rs**2 * SOUND_HORIZON_DRAG_EPOCH_MPC)

# plt.errorbar(z_desi[mask_DH], H_from_desi, yerr=H_err_from_desi, fmt='s', color='k', mfc='none', ms=6, capsize=3, lw=1.0, label='BAO data')

# # Plot LCDM line from astropy
# plt.plot(z_plot, LCDM.H(z_plot).value, color="black", ls="--", lw=1.5, label=r"$\Lambda$CDM")

# # Labels, legend, style
# plt.xlabel(r'$z$')
# plt.ylabel(r'$H(z)$ [km/s/Mpc]')
# plt.xlim(0, max(2.5, z_desi.max()))
# plt.legend(loc='best', frameon=False)
# plt.tight_layout()
# plt.savefig(f'figures/Hz_{MODEL}_{NWALKERS}walkers_{NSTEPS}steps.pdf')
# plt.show()

# =========================================================
# Plotting: Trace Plots
# =========================================================

# def plot_trace(
#     samples_dict,
#     params_to_plot,
#     param_label_map,
#     runs_to_plot,
#     max_walkers=10,   # limit for visibility
#     alpha=0.3
# ):
#     """
#     Plot trace plots for selected parameters and runs.

#     Parameters
#     ----------
#     samples_dict : dict
#         {run_name: samples array}
#     params_to_plot : list
#         List of parameter names (keys of param_label_map)
#     param_label_map : dict
#         Mapping from param name -> LaTeX label
#     runs_to_plot : dict
#         Subset of runs (like HZ_PLOT_RUNS)
#     max_walkers : int
#         Max number of walkers to plot per run
#     alpha : float
#         Line transparency
#     """

#     param_names = list(param_label_map.keys())
#     n_params = len(params_to_plot)

#     fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.5 * n_params), sharex=True)

#     if n_params == 1:
#         axes = [axes]

#     for i, param in enumerate(params_to_plot):
#         ax = axes[i]
#         idx = param_names.index(param)

#         for run_name, info in runs_to_plot.items():
#             samples = samples_dict[run_name]

#             # --- Ensure shape: (n_walkers, n_steps, n_params)
#             if samples.ndim == 2:
#                 # (n_steps, n_params) → fake 1 walker
#                 chain = samples[None, :, :]
#             elif samples.ndim == 3:
#                 chain = samples
#             else:
#                 raise ValueError(f"Unexpected shape: {samples.shape}")

#             n_walkers, n_steps, _ = chain.shape

#             # Limit walkers for clarity
#             walkers_to_plot = min(n_walkers, max_walkers)

#             for w in range(walkers_to_plot):
#                 ax.plot(
#                     chain[w, :, idx],
#                     color=info["colour"],
#                     alpha=alpha,
#                     lw=0.8
#                 )

#         ax.set_ylabel(param_label_map[param])
#         ax.grid(True)

#     axes[-1].set_xlabel("MCMC step")

#     # Legend (only once)
#     handles = [
#         plt.Line2D([0], [0], color=info["colour"], label=info["label"])
#         for info in runs_to_plot.values()
#     ]
#     axes[0].legend(handles=handles)

#     plt.tight_layout()
#     plt.savefig(f'figures/trace_{MODEL}_{NWALKERS}walkers_{NSTEPS}steps.pdf')
#     plt.show()

# plot_trace(
#     samples_dict=samples_dict,
#     params_to_plot=['H0', 'q0', 'j0', 'Om0'],
#     param_label_map=PARAM_LABEL_MAP,
#     runs_to_plot=HZ_PLOT_RUNS
# )

# # =========================================================
# # Plotting: IACT --> MAKE SURE YOU HAVE ENOUGH STEPS
# # =========================================================
# sampler_dict = {}
# params_to_plot = ['H0', 'q0', 'j0', 'Om0']

# # Create 2x2 subplots
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
# axes = axes.flatten()

# param_names = PARAM_ORDER

# for i, param in enumerate(params_to_plot):
#     ax = axes[i]
#     idx = param_names.index(param)

#     for run_name, info in HZ_PLOT_RUNS.items():
#         chain_full = chain_dict[run_name]   # (n_walkers, n_steps, n_params)

#         chain = chain_full[:, :, idx]       # (n_walkers, n_steps)
#         n_walkers, n_steps = chain.shape

#         steps_range = np.linspace(200, n_steps, 50, dtype=int)
#         cum_tau = []

#         for n in steps_range:
#             try:
#                 tau_est = emcee.autocorr.integrated_time(
#                     chain[:, :n],
#                     tol=50,
#                     quiet=True
#                 )
#                 cum_tau.append(tau_est[0])
#             except emcee.autocorr.AutocorrError:
#                 cum_tau.append(np.nan)

#         cum_tau = np.array(cum_tau)
#         mask = np.isfinite(cum_tau)

#         print(run_name, param, np.sum(mask), np.isnan(cum_tau).all())

#         if np.sum(mask) > 5:
#             ax.plot(
#                 steps_range[mask],
#                 cum_tau[mask],
#                 color=info["colour"],
#                 label=info["label"]
#             )

#     ax.set_title(PARAM_LABEL_MAP[param])
#     ax.set_ylabel(r"IACT ($\tau$)")
#     ax.grid(True)

# # Only bottom row gets x-labels
# for ax in axes[-2:]:
#     ax.set_xlabel("MCMC steps")

# # Add legend to first subplot
# axes[0].legend()
# plt.tight_layout()
# plt.savefig(f'figures/IACT_{MODEL}_{NWALKERS}walkers_{NSTEPS}steps.pdf')
# plt.show()

# def effective_sample_size(chain):
#     tau = emcee.autocorr.integrated_time(chain, quiet=True)
#     n_walkers, n_steps = chain.shape
#     return (n_walkers * n_steps) / tau

# chain = chain_dict['planck+union+pantheon+desy'][:, :, 0]  # H0
# Neff = effective_sample_size(chain)
# print("Neff:", Neff)

# =========================================================
# Plotting: w(z)
# =========================================================


