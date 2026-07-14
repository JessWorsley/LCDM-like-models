from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# =========================================================
# Plotting
# =========================================================

IBM_PALETTE = ["#648FFF", '#785EF0', '#DC267F', '#FE6100', '#FFB000']
WONG_PALETTE = ['#CC79A7', '#D55E00', '#E69F00', '#009E73', '#56B4E9', '#0072B2']

PLOT_TEXT_OPTIONS = {
    'text.usetex': True,
    'font.family': 'serif',
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18,
}

# =========================================================
# Mathematial constants
# =========================================================

EXP_THRESHOLD = 700.0

# =========================================================
# Physical constants
# =========================================================

SPEED_OF_LIGHT_KM_S = 299792.458 # km/s
SOUND_HORIZON_DRAG_EPOCH_MPC = 147.2 # r_s(z_d)
DRAG_EPOCH_REDSHIFT = 1060
SOUND_HORIZON_RECOMBINATION_EPOCH_MPC = 144.4 # r_s(z_*)
RECOMBINATION_REDSHIFT = 1089.92
OMEGA_M0 = 0.3
LCDM = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)

# =========================================================
# Models
# =========================================================

MODEL1 = 'model_1'
MODEL2 = 'model_2'
MODEL3 = 'model_3'
CPL = 'cpl'

# =========================================================
# MCMC Parameters
# =========================================================

NWALKERS = 64       # no. chains
NSTEPS = 20000      # no. steps
BURN = 2000         # how many to discard
THIN = 20           # take every nth value
SEED = 42           # set random seed

# =========================================================
# Datasets, labels, colours
# =========================================================

DATASET_DESI = 'desi'
DATASET_DESI_UNION = 'union'
DATASET_DESI_PANTHEON = 'pantheon'
DATASET_DESI_DESY = 'desy'

PARAM_NAMES = ['H0', 'q0', 'j0', 'Om0']
COLOURS = ['C0','C1','C2','C3','C4','C5']

DATASETS = {
    'desi':     {'label': 'DESI',      'colour': 'C1'},
    'union':    {'label': 'Union3',    'colour': 'C2'},
    'pantheon': {'label': 'Pantheon+', 'colour': 'C3'},
    'desy':     {'label': 'DESY5',     'colour': 'C4'},
    'planck':   {'label': 'Planck',    'colour': 'C5'}
}

DATA_PARAMS = {
    'desi':     ['H0','q0','j0','eps','c1'],
    'union':    ['H0','q0','j0','eps','c1'],
    'pantheon': ['H0','q0','j0','eps','c1'],
    'desy':     ['H0','q0','j0','eps','c1'],
    'planck':   ['H0','q0','j0','Om0','eps','c1','c2','wDE0','wDE0p']
}

PARAM_ORDER = ['H0','q0','j0','Om0','eps','c1','c2','wDE0','wDE0p']

PARAM_LABEL_MAP = {
    'H0': r'$H_0$',
    'q0': r'$q_0$',
    'j0': r'$j_0$',
    'Om0': r'$\Omega_{m0}$',
    'eps': r'$\epsilon$',
    'c1': r'$c_1$',
    'c2': r'$c_2$',
    'wDE0': r'$w_0$',
    'wDE0p': r"$w'_0$"
}

HZ_PLOT_RUNS = {
    "union+pantheon+desy": {
        "label": "DESI + Union3 + Pantheon+ + DESY5",
        "colour": "C0"},

    "planck+union+pantheon+desy": {
        "label": "DESI + Planck + Union3 + Pantheon+ + DESY5",
        "colour": "C2"},

    "planck": {
        "label": "DESI + Planck",
        "colour": "C4"}
}