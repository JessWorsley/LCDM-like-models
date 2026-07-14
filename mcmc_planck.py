import matplotlib.pyplot as plt
import numpy as np

from getdist import loadMCSamples
from classy import Class
from constants import SPEED_OF_LIGHT_KM_S, EXP_THRESHOLD, OMEGA_M0

def Hz_from_class(z, H0, omegabh2, omegach2):

    cosmo = Class()

    params = {
        "H0": H0,
        "omega_b": omegabh2,
        "omega_cdm": omegach2,
        "output": "mPk",
        "N_ur": 2.0328,
        "N_ncdm": 1,
        "m_ncdm": 0.06
    }

    cosmo.set(params)
    cosmo.compute()

    Hz = cosmo.Hubble(z) * SPEED_OF_LIGHT_KM_S

    cosmo.struct_cleanup()
    cosmo.empty()

    return Hz

def H_LCDM(z, H0, Om, OL):
    return H0 * (Om*(1+z)**3 + OL)**0.5

def load_planck():
    planck_data_fileloc = 'data/planck/COM_CosmoParams_fullGrid_R3.01/base/plikHM_TTTEEE_lowl_lowE/base_plikHM_TTTEEE_lowl_lowE'
    samples = loadMCSamples(planck_data_fileloc)
    theta = samples.getParams()

    H0 = theta.H0
    omegabh2 = theta.omegabh2
    omegach2 = theta.omegach2

    N_chain = 200
    z = np.linspace(0,3,1000)

    idx = np.random.choice(len(H0), N_chain, replace=False)

    Hz_samples = []

    for i in idx:
        Hz_samples.append(
            Hz_from_class(z, H0[i], omegabh2[i], omegach2[i])
        )

    Hz_samples = np.array(Hz_samples)   # shape (N_chain, Nz)
    Hz_means = Hz_samples.mean(axis=0)
    Hz_cov = np.cov(Hz_samples, rowvar=False)
    Hz_inv_cov = np.linalg.inv(Hz_cov)

    return Hz_means, Hz_cov, Hz_inv_cov, z

def planck_distance_theory(params):
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    z_star = cosmo.get_current_derived_parameters(['z_rec'])['z_rec']

    DM = cosmo.angular_distance(z_star) * (1 + z_star)   # comoving distance
    rs = cosmo.rs_drag()   # sound horizon

    H0 = params["H0"]
    h = H0/100

    Omega_m = (params["omega_b"] + params["omega_cdm"]) / h**2

    R = np.sqrt(Omega_m) * H0 * DM / SPEED_OF_LIGHT_KM_S
    lA = np.pi * DM / rs
    omegabh2 = params["omega_b"]

    cosmo.struct_cleanup()
    cosmo.empty()

    return np.array([R, lA, omegabh2])

def loglike_planck(theta):
    v_obs = np.array([1.74963, 301.80845, 0.02237])

    C = np.array([
        [2.5e-5, 2.2e-4, -2.0e-7],
        [2.2e-4, 0.090, -2.0e-5],
        [-2.0e-7, -2.0e-5, 2.0e-8]
    ])

    Cinv = np.linalg.inv(C)
    params = {
        "H0": theta["H0"],
        "omega_b": theta["omegabh2"],
        "omega_cdm": theta["omegach2"],
        "output": "mPk"
    }

    v_th = planck_distance_theory(params)

    diff = v_th - v_obs
    chi2 = diff @ Cinv @ diff

    return -0.5 * chi2

def background_analytical(theta, z_eval, model='model_1', exp_threshold=EXP_THRESHOLD):
    """
    Analytic H(z) that includes q0 and j0 (eq 34a from Chakraborty, Louw, et al 2025).
    Returns H(z) array or None on failure (invalid params / overflow / negative inside sqrt).
    Set safe threshold: np.exp(700) is ≈ 1e304 (close to float max)
    """
    H0, q0, j0 = theta

    # Guard against infinite values
    if not np.isfinite(H0) or not np.isfinite(q0) or not np.isfinite(j0):
        return None

    # Avoid q0=-1 which makes denominator blow up
    if abs(1.0 + q0) < 1e-12:
        return None
    
    # Compute (1+z)**alpha via exp(exponent * log1p(z)), but guard exponent size
    z_eval = np.asarray(z_eval)
    log_base = np.log1p(z_eval)  # log(1+z) is stable for small z

    if model == 'model_1':
        exponent = (2.0 + j0 + 3.0 * q0) / (1.0 + q0)
        denominator = (2.0 + j0 + 3.0 * q0) / (1.0 + q0)**2

        # avoid tiny denominator
        if abs(denominator) < 1e-12:
            return None

        # Compute max exponent magnitude
        max_exp_arg = np.max(exponent * log_base)
        min_exp_arg = np.min(exponent * log_base)
        # Overflow guard
        if max_exp_arg > exp_threshold:
            return None
        # Avoid extremely small values:
        if min_exp_arg < -1000:
            pass

        h = H0 * np.sqrt( ( 2 * (1 + z_eval)**exponent + (j0 - q0 * (1 + 2 * q0))/(1 + q0)**2 ) / denominator )
        Omega_m = (OMEGA_M0*(z_eval + 1)**3 * (j0 + 3*q0 + 2)) / ( (q0 + 1)**2 * (2*(z_eval + 1)**((j0+3*q0+2)/(q0+1)) - (j0-q0*(2*q0+1))/(q0+1)**2) )
        try:
            return h, Omega_m
        except Exception:
            return None

    else:
        raise ValueError(f"No model named {model}. Please specify 'model_1', 'model_2', or 'model_3'.")