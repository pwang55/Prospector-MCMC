"""
Run prospector+MCMC on a single selected SPHEREx spectra, and save figures/parameter medians & 16, 84 percentiles

Example:
$ python prospector_mcmc.py --config mcmc_config.yaml --spherex_id 1663027938116763658 -f spherex_gals.parq -n 32 -o PLOTS

Notes:
    - a yaml configuration file is required as an input in CLI mode
    - any following keywords overrides the configuration settings
    - by default, this script saves MCMC settings/metadata in "mcmc_results_{spherex_id}.yaml" 
      and MCMC key ressults in "mcmc_results_{spherex_id}.npz"
    - to check available override keywords, use -h
"""
import os
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml
import emcee
import multiprocessing
from scipy.stats import norm, t, lognorm, loguniform
from functools import partial
import custom_prospector_tools as cpt
from types import SimpleNamespace
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from pathlib import Path
import corner
from IPython.display import display, Math
# from IPython import get_ipython
import sedpy
import argparse
import gc
import time
from datetime import datetime
import h5py

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default=None,
        metavar='<str>',
        help='Configuration YAML file',
        required=True,
    )
    parser.add_argument(
        '-si',
        '--spherex_id',
        type=int,
        default=None,
        metavar='<int>',
        help='SPHERExRefID, will override the id saved in config file'
    )
    parser.add_argument(
        '-f',
        '--filename',
        type=str,
        default=None,
        metavar='<str>',
        help='SPHEREx catalog name'
    )
    parser.add_argument(
        '-fl',
        '--filter_list',
        type=str,
        default=None,
        metavar='<str>',
        help='SPHEREx fiducial_filters.txt path'
    )
    parser.add_argument(
        '-nw',
        '--nwalkers',
        type=int,
        default=None,
        metavar='<int>',
        help='number of emcee ensemble walkers'
    )
    parser.add_argument(
        '-j',
        '--jitter',
        type=float,
        default=None,
        metavar='<float>',
        help='random jitter from MCMC initial position'
    )
    parser.add_argument(
        '-ns',
        '--nsteps',
        type=int,
        default=None,
        metavar='<int>',
        help='number of emcee chain steps'
    )
    parser.add_argument(
        '-d',
        '--discard',
        type=int,
        default=None,
        metavar='<int>',
        help='number of emcee steps to discard'
    )
    parser.add_argument(
        '-t',
        '--thin',
        type=int,
        default=None,
        metavar='<int>',
        help='number of steps to skip to thin the chains'
    )
    parser.add_argument(
        '-nzp',
        '--no_zprior',
        action='store_true',
        default=None,
        help="Don't use photoz and sigma_photz as gaussian prior for zred"
    )
    parser.add_argument(
        '-p',
        '--parallel',
        action='store_true',
        default=None,
        help='Use python multiprocessing'
    )
    parser.add_argument(
        '-ss',
        '--save_sampler',
        action='store_true',
        default=None,
        help='Save sampler to a .h5 file'
    )
    parser.add_argument(
        '-sf',
        '--sampler_filename',
        type=str,
        default=None,
        metavar='<str>',
        help='If --save_sampler, save sampler backend to this filename (if "use_id", filename will be "mcmc_results_sampler_[spherex_id].h5")'
    )
    parser.add_argument(
        '-o',
        '--output_filename',
        type=str,
        default=None,
        metavar='<str>',
        help='Output report and results file name (if "use_id", filename will be "mcmc_results_[spherex_id].h5")'
    )
    parser.add_argument(
        '-np',
        '--no_plots',
        action='store_true',
        default=None,
        help="Don't save plots to plots_dir"
    )
    parser.add_argument(
        '-pd',
        '--plots_dir',
        type=str,
        default=None,
        metavar='<str>',
        help='Output plots directory'
    )
    parser.add_argument(
        '-od',
        '--output_dir',
        type=str,
        default=None,
        metavar='<str>',
        help='Output report file and mcmc result directory")'
    )
    parser.add_argument(
        '-q', 
        '--quiet', 
        action='store_true',
        default=None,
        help='Silence the verbose outputs'
    )
    return parser.parse_args()

# ===========================================
# Default settings class
# ===========================================

# default cosmology
cosmology = Planck18

class default_settings:
    """
    Provide some common default/initial settings for the script to read from
    Change this section if we want to change any default/starting values for parameters.
    """
    def __init__(self):
        # TEMP
        self.nwalkers = 32
        self.jitter = 1e-4
        self.nsteps = 10000
        self.discard = 2000
        self.thin = 10
        self.discard_fraction = 5   # if discard > nsteps, use nsteps/discard_fraction as the new discard

    def initial_vals(self, myparams=None):
        """
        This includes all possible free parameters in all models, but not all will be used
        """
        try:
            n_logsfr_ratios = myparams['nbins']-1
        except:
            n_logsfr_ratios = 6
        dict1 = {
            'zred': 0.3,
            'logzsol': 0.0,
            'logmass': 10,
            'dust2': 1.0,
            'dust_ratio': 1.0,
            'dust_index': -1.0,
            'duste_loggamma': -2,
            'duste_gamma': 0.01,
            'duste_umin': 1.0,
            'duste_qpah': 3.5,
            'gas_logz': 0.0,
            'gas_logu': -2.0,
            'fagn': 1e-4,
            'agn_tau': 5.0,
            'logsfr_ratios': [0.0]*n_logsfr_ratios,
            'tage_tuniv': 0.9,
            'tau': 2,
            'fage_trunc': 0.95,
            'sf_slope': -0.5,
            'fburst': 0.1,
            'fage_burst': 0.9
        }
        return dict1
    
    def free_params(self):
        """
        Pre-defined free parameters if user doesn't specify
        """
        default_params = ['zred', 
                          'logzsol',
                          'logmass',
                          'dust2',
                          'dust_ratio',
                          'dust_index',
                          'duste_gamma',
                          'duste_umin',
                          'duste_qpah',
                          'logsfr_ratios',
                          'tage_tuniv',
                          'tau',
                          'fage_trunc',
                          'sf_slope',
                          'fburst',
                          'fage_burst']
        return default_params

    def default_priors(self, redshift=0.1, redshift_sigma=0.1):
        # use partial to pre-fill logpdf functions so that they can be easily called in log_prior
        prior_funcs = {
            'zred':             partial(norm_logpdf_cutoffs, loc=redshift, scale=redshift_sigma, low=0.001, high=3.0),
            'logzsol':          partial(uniform_logpdf, low=-2.0, high=0.4),
            'logmass':          partial(uniform_logpdf, low=6, high=12),
            'logsfr_ratios':    partial(t_logpdf_cutoffs, low=-5, high=5, df=2, loc=0, scale=0.3),
            # 'dust2':            partial(uniform_logpdf, low=0.0, high=3.0),
            'dust2':            partial(norm_logpdf_cutoffs, loc=0.3, scale=1, low=0.0, high=4.0),
            # 'dust_ratio':       partial(uniform_logpdf, low=0.0, high=2.0),
            'dust_ratio':       partial(norm_logpdf_cutoffs, loc=1, scale=0.3, low=0.0, high=2.0),
            'dust_index':       partial(uniform_logpdf, low=-1.3, high=0.4),
            # 'duste_gamma':      partial(lognorm_logpdf_cutoffs, norm_mean=-2, norm_sig=1, low=1e-4, high=1.0),
            'duste_gamma':      partial(uniform_logpdf, low=1e-4, high=1.0),
            'duste_loggamma':   partial(uniform_logpdf, low=-4.0, high=0.0),
            'duste_umin':       partial(norm_logpdf_cutoffs, loc=1., scale=10., low=0.1, high=25.0),
            'duste_qpah':       partial(norm_logpdf_cutoffs, loc=2., scale=2., low=0.0, high=7.0),
            'gas_logz':         partial(uniform_logpdf, low=-2.0, high=0.4),
            'gas_logu':         partial(uniform_logpdf, low=-4.0, high=-1.0),
            'fagn':             partial(loguniform.logpdf, a=1e-4, b=3),
            'log_fagn':         partial(uniform_logpdf, low=-4, high=np.log10(3)),
            'agn_tau':          partial(loguniform.logpdf, a=5.0, b=150.),
            'log_agn_tau':      partial(uniform_logpdf, low=np.log10(5.0), high=np.log10(150.)),
            'tage_tuniv':       partial(uniform_logpdf, low=0.001, high=1),
            'tau':              partial(loguniform.logpdf, a=0.1, b=30),
            'fage_trunc':       partial(uniform_logpdf, low=0.5, high=1.0),
            'sf_slope':         partial(uniform_logpdf, low=-10.0, high=10.0),
            'fburst':           partial(uniform_logpdf, low=0.0, high=0.5),
            'fage_burst':       partial(uniform_logpdf, low=0.5, high=1.0),
        }        

        return prior_funcs



# create the default object for the scripts to read as global variable
global_defaults = default_settings()


# ===========================================
# Catalog and data handling
# ===========================================

filters_ls = sedpy.observate.load_filters(['bass_g', 'bass_r', 'mzls_z'])
filters_ps1 = sedpy.observate.load_filters(['panstarrs_g', 'panstarrs_r', 'panstarrs_i', 'panstarrs_z', 'panstarrs_y'])
filters_2mass = sedpy.observate.load_filters(['twomass_J', 'twomass_H', 'twomass_Ks'])
filters_wise = sedpy.observate.load_filters(['wise_w1', 'wise_w2', 'wise_w3', 'wise_w4'])

lbs_ls = np.array([filters_ls[i].wave_effective for i in range(len(filters_ls))])/10000
lbs_ps1 = np.array([filters_ps1[i].wave_effective for i in range(len(filters_ps1))])/10000
lbs_2mass = np.array([filters_2mass[i].wave_effective for i in range(len(filters_2mass))])/10000
lbs_wise = np.array([filters_wise[i].wave_effective for i in range(len(filters_wise))])/10000
external_phot_lbs = {
    'LS': lbs_ls,
    'PS1': lbs_ps1,
    '2MASS': lbs_2mass,
    'WISE': lbs_wise
}

# Reference catalog external photometry column names
LS_cols = ['LS_g', 'LS_r', 'LS_z']
PS1_cols = ['PS1_g', 'PS1_r', 'PS1_i', 'PS1_z', 'PS1_y']
twomass_cols = ['2MASS_J', '2MASS_H', '2MASS_Ks']
wise_cols = ['WISE_W1', 'WISE_W2', 'WISE_W3', 'WISE_W4']
all_refcat_surveys = [LS_cols, PS1_cols, twomass_cols, wise_cols]

class catalog:
    """
    Read input SPHEREx data format and convert to useful arrays

    Attributes
    ----------
    dat : pandas.DataFrame
        Full input data table
    spherex_ids : ndarray
        All SPHEREx unique reference catalog ids in this table
    zspecs : ndarray
        All spectroscopic redshifts
    zphots : ndarray
        All photometric redshifts from L4 catalog
    zphots_u68 : ndarray
        All upper 1-sigma photo-zs
    zphots_l68 : ndarray
        All lower 1-sigma photo-zs
    zphots_std : ndarray
        All standard deviation of photo-z pdfs
    spectra : ndarray of shape (nrows, nfilts)
        All spectra in this table
    error : ndarray of shape (nrows, nfilts)
        All uncertainty in this table
    frac102 : ndarray
        All frac102 values in this table

    Methods
    -------
    get_external_phots(SPHERExRefID, idx)
        Grab all external photometry for a given source, whether with SPHERExRefID or row index
        Returns a nested dictionary with 'LS', 'PS1', '2MASS', 'WISE', 
        each with keys 'wavelength', 'flux', 'flux_error'
    """
    def __init__(self, filename=''):
        self.dat = pd.read_parquet(filename)
        self.spherex_ids = np.array(self.dat['SPHERExRefID'])
        self.zspecs = np.array(self.dat['z_specz'])
        self.zphots = np.array(self.dat['z_best_gals'])
        self.zphots_u68 = np.array(self.dat['z_err_u68_gals'])
        self.zphots_l68 = np.array(self.dat['z_err_l68_gals'])
        self.zphots_std = np.array(self.dat['z_err_std_gals'])
        self.spectra = np.stack(self.dat['flux_dered_fiducial'])
        self.error = np.stack(self.dat['flux_err_dered_fiducial'])
        self.frac102 = np.array(self.dat['frac_sampled_102'])

    def get_external_phots(self, SPHERExRefID=None, idx=None):
        if SPHERExRefID is not None:
            idx = np.where(self.spherex_ids == SPHERExRefID)[0][0]

        external_phots = {}
        for i, survey_cols in enumerate(all_refcat_surveys):
            ndat = len(survey_cols)
            survey_name = survey_cols[0].split('_')[0]
            wavelength = external_phot_lbs[survey_name]
            flux = np.zeros(ndat)
            flux_error = np.zeros(ndat)
            for j, colname in enumerate(survey_cols):
                flux[j] = self.dat[colname].iloc[idx]
                flux_error[j] = self.dat[colname+'_error'].iloc[idx]
            external_phots[survey_name] = {
                                            'wavelength':wavelength,
                                            'flux': flux,
                                            'flux_error': flux_error
                                            }
        return external_phots

class catalog_dataset:
    """
    Use pyarrow.dataset to access SPHEREx L4 Catalog

    Parameters
    ----------
    filename : str
        L4 parquet catalog filename
    
    Methods
    -------
    get_row(SPHERExRefID) : 
        Get useful information from a single row with SPHERExRefID
        Creates

    """
    def __init__(self, filename):
        self.dataset = ds.dataset(filename, format='parquet')

    def get_row(self, SPHERExRefID):
        ds_filters = ds.field('SPHERExRefID') == SPHERExRefID
        tab = self.dataset.to_table(filter=ds_filters)
        self.zspec = tab['z_specz'][0].as_py()
        self.zphot = tab['z_best_gals'][0].as_py()
        self.zphot_u68 = tab['z_err_u68_gals'][0].as_py()
        self.zphot_l68 = tab['z_err_l68_gals'][0].as_py()
        self.zphot_std = tab['z_err_std_gals'][0].as_py()
        self.frac102 = tab['frac_sampled_102'][0].as_py()
        self.spec = tab['flux_dered_fiducial'].to_numpy()[0]
        self.err = tab['flux_err_dered_fiducial'].to_numpy()[0]

        self.external_phots = {}
        for i, survey_cols in enumerate(all_refcat_surveys):
            ndat = len(survey_cols)
            survey_name = survey_cols[0].split('_')[0]
            wavelength = external_phot_lbs[survey_name]
            flux = np.zeros(ndat)
            flux_error = np.zeros(ndat)
            for j, colname in enumerate(survey_cols):
                flux[j] = tab[colname][0].as_py()
                flux_error[j] = tab[colname+'_error'][0].as_py()
            self.external_phots[survey_name] = {
                                            'wavelength':wavelength,
                                            'flux': flux,
                                            'flux_error': flux_error
                                            }


def save_h5_results(mcmc_results, 
                    output_filename='mcmc_results_report.h5', 
                    output_dir='.',
                    metadata=None):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compressed_cols = ['flat_samples', 'flat_mfracs', 'med_lbs', 'med_spectra']
    
    with h5py.File(Path(output_dir) / output_filename, 'w') as f:
        for i, key in enumerate(mcmc_results):
            if type(mcmc_results[key]) is not dict:
                if key not in compressed_cols:
                    dset = f.create_dataset(key, data=mcmc_results[key])
                else:
                    dset = f.create_dataset(key, data=mcmc_results[key], compression='gzip')
            else:
                group = f.create_group(key)
                for j, keyj in enumerate(mcmc_results[key]):
                    dset2 = group.create_dataset(keyj, data=mcmc_results[key][keyj])

        if metadata is not None:
            for i, key in enumerate(metadata):
                f.attrs[key] = metadata[key]



def read_h5_results(h5file):
    """
    Convenience function to read back the saved mcmc_results_report.h5 file
    Returns the mcmc_results dictionary, and add "attribute" key to the dictionary that includes metadata for the MCMC run
    """
    mcmc_results = {}
    with h5py.File(h5file, 'r') as f:
        for i, key in enumerate(f):
            if key != 'med_myparam':
                mcmc_results[key] = f[key][()]
            else:
                mcmc_results[key] = {}
                for j, keyj in enumerate(f['med_myparam']):
                    mcmc_results[key][keyj] = f['med_myparam'][keyj][()]
        mcmc_results['attributes'] = {}
        for i, key in enumerate(f.attrs) :
            mcmc_results['attributes'][key] = f.attrs[key]
    return mcmc_results

# ===========================================
# experimental multiprocessing functions
# ===========================================

# initializer for python multiprocessing
worker_prospector = None
def initialize_prospector(sfh_type, filters):
    global worker_prospector
    if worker_prospector is None:
        worker_prospector = cpt.custom_prospector(sfh_type=sfh_type, filters=filters)
# intermediate check to see if child process "forget" about prospector, and if it does,
# re-initialize prospector
def get_worker_prospector(sfh_type, filters):
    global worker_prospector
    if worker_prospector is None:
        worker_prospector = cpt.custom_prospector(sfh_type=sfh_type, filters=filters)
    return worker_prospector

# global log probability for multiprocessing
def global_log_probability(theta, flux, flux_error, redshift, redshift_sigma, sfh_tyle, filters, lp_func, myparam_func):
    lp = lp_func(theta, redshift, redshift_sigma)
    worker_myparam = myparam_func(theta)
    if not np.isfinite(lp):
        log_prob = lp
    else:
        worker_prospector_here = get_worker_prospector(sfh_tyle, filters)
        _, _, model_flux, _ = worker_prospector_here.generate_spectra(myparams=worker_myparam)
        ll = -0.5 * np.sum((model_flux-flux)**2/flux_error**2 + np.log(2*np.pi*flux_error**2))
        log_prob = lp + ll
    return log_prob

# ===========================================
# logpdf functions
# ===========================================

def uniform_logpdf(val, low=0.0, high=1.0):
    """
    Return uniform logpdf at val, return 0.0 for low <= val <= high
    """
    if float(low) <= val <= float(high):
        return 0.0
    else:
        return -np.inf
    
def norm_logpdf_cutoffs(val, loc, scale, low, high):
    """
    Return gaussian logpdf at val, for gaussian with mean=loc and std=scale,
    with cutoffs at low and high (inclusive)

    Args:
        val (float): The value at which to evaluate the log-pdf
        loc (float): mean of the gaussian distribution
        scale (float): standard deviation of the gaussian
        low (float): cutoff lower bound
        high (float): cutoff upper bound
    """
    if loc is None or scale is None:
        return uniform_logpdf(val, low=low, high=high)
    else:
        return norm.logpdf(val, loc=loc, scale=scale) + uniform_logpdf(val, low=low, high=high)
    
def lognorm_logpdf_cutoffs(val, low, high, s=None, scale=None, norm_mean=None, norm_sig=None):
    """
    Return lognormal logpdf at val, for base 10 lognormal distribution 
    that is a gaussian with mean=loc and std=scale in log space.
    The funciton automatically calculate spread and scale of the lognormal function.
    Ctoffs at low and high (inclusive)

    Args:
        val (float): The value at which to evaluate the log-pdf
        low (float): cutoff lower bound
        high (float): cutoff upper bound
        norm_mean (float): Mean of the gaussian in log10(val) space
        norm_scale (float): Standard deviation of the gaussian in log10(val) space
        s (float): Alternatively, lognormal shape parameter, s=np.log(10) * norm_scale
        scale (float): Alternatively, lognormal scale parameter, scale=10**(norm_mean)
    """
    if norm_mean is not None and norm_sig is not None:
        s = norm_sig * np.log(10)
        scale = 10**(norm_mean)
    return lognorm.logpdf(val, s=s, scale=scale) + uniform_logpdf(val, low=low, high=high)

def t_logpdf_cutoffs(val, low=-5, high=5, df=2, loc=0, scale=0.3):
    """
    Scipy.stats.t distribution (Student's t) with cutoffs low/high

    Args:
        val (float): The value at which to evaluate the log-pdf
        low (float): cutoff lower bound
        high (float): cutoff upper bound
        df (int): Degrees of freedom for the Student's t-distribution
        loc (float): mean of the distribution
        scale (float): The scale parameter used to stretch or compress the distribution

    Returns:
        float: The log-probability density value evaluated at `val`.
    """
    return t.logpdf(val, df=df, loc=loc, scale=scale) + uniform_logpdf(val, low=low, high=high)


# ===========================================
# Plotting functions
# ===========================================


def plot_chain(samples,
               figsize=(10, 8),
               ylabels=None,
               nrowcol=None,
               save=False,
               filename='mcmc_results_chain.png',
               output_dirname='',
               dpi=300,
               **kwargs):
    """
    From a input full MCMC sample, plot chains for each parameter

    Parameters
    ----------
    figsize : tuple
    ylabels : list of str
        Each parameter names
    nrowcol : tuple/list/ndarray
        Custom number of rows and cols
    save : bool
        Whether to save the plot or not
    filename : str
        Output plot file name, if save=True
    output_dirname : str
        Save plot in this directory
    dpi : int
        Saved plot dpi
    **kwargs : 
        Any matplotlib.pyplot.plot() supported arguments
    """
    ndim = samples.shape[2]
    output_dir = Path(output_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault('linewidth', 0.7)
    kwargs.setdefault('alpha', 0.1)
    kwargs.setdefault('color', 'k')
    if ylabels is None:
        ylabels = [f'theta{i}' for i in range(ndim)]
    if nrowcol is None:
        nrows = int(np.ceil(np.sqrt(ndim))) # calculate the nrow as if the plot is square, take the smallest number that accommodate it
        ncols = int(np.ceil(ndim / nrows))  # use the above nrows, calculate the resulting ncols and round up
    else:
        nrows = nrowcol[0]
        ncols = nrowcol[1]
        if nrows * ncols < ndim:
            raise ValueError(f'Figure with ({nrows},{ncols}) subplots are not enough for {ndim} parameters')

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    for i in range(nrows*ncols):
        row = i % nrows
        col = i // nrows
        ax = axs[row, col]
        if i < ndim:
            ax.plot(samples[:,:,i], **kwargs)
            ax.grid(alpha=0.2)
            ax.set_ylabel(ylabels[i])
        else:
            ax.set_visible(False)
    fig.tight_layout()
    if save:
        plt.savefig(output_dir / filename, dpi=dpi)


def plot_corner(flat_samples,
                ylabels=None,
                figsize=(14,14),
                ticklabelsize=5,
                save=False,
                filename='mcmc_results_corner.png',
                output_dirname='', 
                dpi=300,
                **kwargs):
    """
    From a MCMC flat sample, plot corner plots

    Parameters
    ----------
    flat_samples : ndarray
        Flattened MCMC samples from emcee
    ylabels : list of str
        Each parameter names
    figsize : tuple
        Adjust figure size
    ticklabelsize : int
        Adjust each subplots' tick number size
    save : bool
        Whether to save the plot
    filename : str
        If save=True, the filename to save plot to
    output_dirname : str
        Directory to save the plot to
    dpi : int
        Saved plot dpi
    **kwargs : 
        Any corner.corner() supported arguments
    """
    output_dir = Path(output_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs.setdefault('label_kwargs', {"fontsize": 6.5})
    kwargs.setdefault('contour_kwargs', {"colors": 'k', "linewidths": 0.7, 'alpha': 0.7})
    kwargs.setdefault('color', 'k')
    kwargs.setdefault('show_titles', True)
    kwargs.setdefault('title_kwargs', {"fontsize": 5})
    kwargs.setdefault('quantiles', [0.16, 0.5, 0.84])   # corner.py use quantile to describe percentile?
    # contour keyword, data_kwargs, contourf_kwargs etc

    ndim = flat_samples.shape[1]
    if ylabels is None:
        ylabels = [f'theta{i}' for i in range(ndim)]
    fig_corner = plt.figure(figsize=figsize)
    corner.corner( 
        flat_samples, 
        labels=ylabels, 
        fig=fig_corner, 
        **kwargs
    )
    for ax in fig_corner.get_axes():
        ax.tick_params(axis='both', labelsize=ticklabelsize)

    if save:
        plt.savefig(output_dir / filename, dpi=dpi)



def plot_sed_sfh(lamb_obs,
                 spec_obs,
                 err_obs,
                 lamb_model,
                 spec_model,
                 agelims_model=None,
                 sfrsteps_model=None,
                 qs_agelims=None,
                 qs_sfrsteps=None,
                 tl_model=None,
                 sfr_model=None,
                 tl_burst_model=None,
                 qs_tl=None,
                 qs_sfrs=None,
                 qs_tl_burst=None,
                 external_phots=None,
                 figsize=(8,8),
                 sed_obs_kwargs=None,
                 sed_model_kwargs=None,
                 sed_external_phot_kwargs=None,
                 sfh_kwargs=None,
                 sfh_range_kwargs=None,
                 qs_tl_burst_kwargs=None,
                 save=False,
                 xscale='log',
                 yscale='log',
                 filename='mcmc_results_sed_sfh.png',
                 output_dirname='', 
                 dpi=300,
                 title_kwargs=None
                 ):
    """
    Plot data points, model and external photometry
    Optionally plot Star Formation History. All ages/lookback times have to have consistent unit (Gyr or yr)

    Parameters
    ----------
    lamb_obs : ndarray
        Observed spectra wavelength points
    spec_obs : ndarray
        Observed spectra
    err_obs : ndarray
        Observed spectra uncertainties
    lamb_model : ndarray
        Prospecter output spectra wavelength grid
    spec_model : ndarray
        Prospector output model spectra
    agelims_model : ndarray of shape (nbins, 2) or (nbins+1, )
        For continuity_sfh
        Prospector output agebins or 1-d age limits including start and end
        (nbins+1) is for plt.step compatible format
    sfrsteps_model : ndarray of shape (nbins, ) or (nbins+1, )
        For continuity_sfh
        If agebins are provided above, use SFR for each bin
        If agelims are provided above, use SFR with last element repeated (for plt.step format)
    qs_agelims : ndarray of shape (nsfr_step, )
        For continuity_sfh
        Common age limits for all SFR percentiles
    qs_sfrsteps : ndarray of shape (2, nsfr_step) or (3, nsfr_step)
        For continuity_sfh
        Percentiles of SFH from continuity_sfh_percentiles_steps(), has to be plt.step format (last element for each SFR has to repeat)
        If shape[0] = 3, treat the array as [16, 50, 84]
    tl_model : ndarray
        For parametric_sfh
        lookback time for model SFR
    sfr_model : ndarray
        For parametric_sfh
        Parametric SFH output SFR at each lookback time
    tl_burst_model : float
        For parametric_sfh
        Burst lookback time
    qs_tl : ndarray of shape (nsfr, )
        For parametric_sfh
        Common lookback time grid for SFH percentiles
    qs_sfrs : ndarray of shape (2, nsfr) or (3, nsfr)
        For parametric_sfh
        SFR percentiles with respect to the qs_tl grid
    external_phots : dict
        Dictionary of external photometry from refcat, from catalog.get_external_phots(SPHERExRefID)
    figsize : tuple
        Adjust figure size
    save : bool
        Whether to save the plot
    filename : str
        If save=True, the filename to save plot to
    output_dirname : str
        Directory to save the plot to
    dpi : int
        Saved plot dpi
    xscale/yscale : str
        can be 'linear' or 'log'
    sed_obs_kwargs : dict
        Arguments for ax.errorbar()
    sed_model_kwargs : dict 
        Arguments for ax.plot()
    sed_external_phot_kwargs : dict 
        Arguments for ax.errorbar()
    sfh_kwargs : dict 
        Arguments for ax.step()
    sfh_range_kwargs : dict 
        Arguments for ax.fill_between()
    qs_tl_burst_kwargs : dict 
        Arguments for ax.axvspan()
    title_kwargs : dict 
        Arguments for ax.set_title()
    """
    # age_factor = 1.0

    output_dir = Path(output_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)

    sed_obs_kwargs = sed_obs_kwargs or {}
    sed_model_kwargs = sed_model_kwargs or {}
    sed_external_phot_kwargs = sed_external_phot_kwargs or {}
    sfh_kwargs = sfh_kwargs or {}
    sfh_range_kwargs = sfh_range_kwargs or {}
    qs_tl_burst_kwargs = qs_tl_burst_kwargs or {}
    external_phots = external_phots or {}

    # set up default plotting styles
    sed_obs_kwargs.setdefault('fmt', 'o')
    sed_obs_kwargs.setdefault('elinewidth', 1)
    sed_obs_kwargs.setdefault('markersize', 3)
    sed_obs_kwargs.setdefault('markerfacecolor', 'none')
    sed_obs_kwargs.setdefault('markeredgecolor', 'tab:blue')
    sed_obs_kwargs.setdefault('color', 'tab:blue')
    sed_obs_kwargs.setdefault('markeredgewidth', 1.5)
    sed_obs_kwargs.setdefault('capsize', 2)
    sed_obs_kwargs.setdefault('alpha', 0.7)

    sed_model_kwargs.setdefault('linewidth', 1)
    sed_model_kwargs.setdefault('color', 'tab:orange')
    sed_model_kwargs.setdefault('alpha', 0.9)

    sed_external_phot_kwargs.setdefault('fmt', 'o')
    sed_external_phot_kwargs.setdefault('elinewidth', 1)
    sed_external_phot_kwargs.setdefault('markersize', 5)
    sed_external_phot_kwargs.setdefault('markerfacecolor', 'none')
    # sed_external_phot_kwargs.setdefault('markerfacecolor', 'o')
    sed_external_phot_kwargs.setdefault('markeredgewidth', 1.5)
    sed_external_phot_kwargs.setdefault('capsize', 2)
    sed_external_phot_kwargs.setdefault('alpha', 0.5)

    sfh_kwargs.setdefault('color', 'tab:orange')
    sfh_kwargs.setdefault('linewidth', 1)
    sfh_kwargs.setdefault('alpha', 1)

    sfh_range_kwargs.setdefault('color', 'tab:blue')
    sfh_range_kwargs.setdefault('step', 'post')
    sfh_range_kwargs.setdefault('alpha', 0.2)

    qs_tl_burst_kwargs.setdefault('color', 'tab:green')
    qs_tl_burst_kwargs.setdefault('alpha', 0.1)

    nrow = 2
    ncol = 1
    if agelims_model is sfrsteps_model is qs_agelims is qs_sfrsteps is tl_model is sfr_model is qs_tl is qs_sfrs is None:
        # if SFHs are not provided just plot SED
        figsize = (8,4)
        nrow = 1
    
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    for i, axi in enumerate(fig.get_axes()):
        if i == 0:
            # plot SED
            nonzeros = err_obs != 50000
            axi.errorbar(lamb_obs[nonzeros],
                            spec_obs[nonzeros],
                            err_obs[nonzeros],
                            **sed_obs_kwargs,
                            label='Observed spectra')

            axi.plot(lamb_model,
                        spec_model,
                        **sed_model_kwargs,
                        label='Medium model')
            axi.grid()
            lbs_mins = [np.min(lamb_obs)]
            lbs_maxs = [np.max(lamb_obs)]
            if np.max(lamb_model)>7:
                axi.set_xscale('log')
            if external_phots:
                # Two dummy plots to cycle the color to tab:green
                axi.plot([],[])
                axi.plot([],[])
                for key, each_phot_dict in external_phots.items():
                    axi.errorbar(each_phot_dict['wavelength'],
                                    each_phot_dict['flux'],
                                    each_phot_dict['flux_error'],
                                    **sed_external_phot_kwargs,
                                    label=key)
                    lbs_mins.append(np.min(each_phot_dict['wavelength']))
                    lbs_maxs.append(np.max(each_phot_dict['wavelength']))

            axi.legend()
            xmin = np.min(lbs_mins) * 0.8
            xmax = np.max(lbs_maxs) * 1.2
            axi.set_xlim(xmin, xmax)
            if title_kwargs is not None:
                title_strings = []
                spherex_id = title_kwargs.get('spherex_id')
                zspec = title_kwargs.get('zspec')
                zphot = title_kwargs.get('zphot')
                zphot_u68 = title_kwargs.get('zphot_u68')
                zphot_l68 = title_kwargs.get('zphot_l68')
                zmcmc_16 = title_kwargs.get('zmcmc_16')
                zmcmc_med = title_kwargs.get('zmcmc_med')
                zmcmc_84 = title_kwargs.get('zmcmc_84')
                frac102 = title_kwargs.get('frac102')
                fontsize = title_kwargs.get('fontsize')
                if spherex_id is not None:
                    title_strings.append(f"SPHERExRefID={spherex_id}")
                if zspec is not None:
                    title_strings.append(fr"$z_{{spec}}$={zspec:.4f}")
                if zphot is not None:
                    if zphot_u68 is not None and zphot_l68 is not None:
                        title_strings.append(fr"$z_{{phot}}={zphot:.3f}^{{+{(zphot_u68-zphot):.3f}}}_{{-{(zphot-zphot_l68):.3f}}}$")
                    else:
                        title_strings.append(fr"$z_{{phot}}={zphot:.3f}$")
                if zmcmc_med is not None:
                    if zmcmc_84 is not None and zmcmc_16 is not None:
                        title_strings.append(fr"$z_{{mcmc}}={zmcmc_med:.3f}^{{+{(zmcmc_84-zmcmc_med):.3f}}}_{{-{(zmcmc_med-zmcmc_16):.3f}}}$")
                    else:
                        title_strings.append(fr"$z_{{mcmc}}={zmcmc_med:.3f}$")
                if frac102 is not None:
                    title_strings.append(f"frac102={frac102:.3f}")
                if fontsize is None:
                    fontsize = 10
                title = ', '.join(title_strings)
                axi.set_title(title, fontsize=fontsize)

            axi.set_xlabel(r'wavelength [$\mu m$]')
            axi.set_ylabel(r'Flux [$\mu m$]')
        if i == 1:
            # if i loop to 1, plot SFH
            if agelims_model is not None: # if continuity sfh is provided
                # check if agelims_model is agebins from prospector or already converted to step required agelims
                if agelims_model.ndim == 2:
                    agelims_model = np.hstack([agelims_model[:,0], agelims_model[-1,1]])
                    sfrsteps_model = np.hstack([sfrsteps_model, sfrsteps_model[-1]])
                # check first agelims are in Gyr
                # convert to Gyr in plottings
                if agelims_model[0] == 1:
                    agelims_model = agelims_model/1e9
                axi.step(agelims_model, 
                         sfrsteps_model, 
                         where='post', 
                         **sfh_kwargs, 
                         label='median parameter SFH')
                if qs_agelims is not None and qs_sfrsteps is not None:
                    if qs_agelims[0] == 1:
                        qs_agelims = qs_agelims/1e9
                    if qs_sfrsteps.shape[0] == 2:
                        idx_84 = 1
                    elif qs_sfrsteps.shape[0] == 3:
                        idx_84 = 2
                        # if 50 percentile SFH exist, plot it
                        sfh_median_kwargs = sfh_kwargs.copy()
                        sfh_median_kwargs['color'] = 'tab:green'
                        axi.step(qs_agelims,
                                 qs_sfrsteps[1],
                                 where='post',
                                 **sfh_median_kwargs,
                                 label='Median SFH from each agebin')
                    axi.fill_between(qs_agelims,
                                     qs_sfrsteps[0],
                                     qs_sfrsteps[idx_84],
                                     **sfh_range_kwargs,
                                     label='68 credible inverval from each agebin')
                axi.set_xlim(np.max(agelims_model), 0.01)
                    
            # if parametric_sfh keywords are given
            elif tl_model is not None:
                if np.median(tl_model) > 1000:  # if it's a big number assuming its in yr
                    tl_model = tl_model/1e9
                axi.plot(tl_model, sfr_model, **sfh_kwargs, label='median parameter SFH')
                if qs_tl is not None and qs_sfrs is not None:
                    if np.median(qs_tl) > 1000:
                        qs_tl = qs_tl/1e9
                        # qs_sfrs = qs_sfrs/1e9
                    if qs_sfrs.shape[0] == 2:
                        idx_84 = 1
                    elif qs_sfrs.shape[0] == 3:
                        idx_84 = 2
                        # if 50 percentile SFH exist, plot it
                        sfh_median_kwargs = sfh_kwargs.copy()
                        sfh_median_kwargs['color'] = 'tab:green'
                        axi.plot(qs_tl, qs_sfrs[1], **sfh_median_kwargs, label=r'Mediah SFH from each $t_{lookback}$')
                    axi.fill_between(qs_tl, qs_sfrs[0], qs_sfrs[2], **sfh_range_kwargs, label=r'68% credible inverval from each $t_{lookback}$')
                    axi.set_xlim(np.max(qs_tl), 0.01)
                ymin, ymax = axi.get_ylim()
                if tl_burst_model is not None:
                    if tl_burst_model > 1000:
                        tl_burst_model = tl_burst_model/1e9
                    axi.vlines(tl_burst_model, ymin=ymin, ymax=ymax, **sfh_kwargs, linestyle='--', label=r'median parameter $tburst_{lookback}$')
                if qs_tl_burst is not None:
                    if np.median(qs_tl_burst) > 1000:
                        qs_tl_burst = qs_tl_burst/1e9
                    if qs_tl_burst.shape[0] == 2:
                        idx_84 = 1
                    elif qs_tl_burst.shape[0] == 3:
                        idx_84 = 2
                        axi.vlines(qs_tl_burst[1], ymin=ymin, ymax=ymax, **sfh_median_kwargs, linestyle='--', label=r'Median $tburst_{lookback}$')
                    axi.axvspan(qs_tl_burst[0], qs_tl_burst[idx_84], **qs_tl_burst_kwargs, label=r'68% cerdible tburst')

            axi.grid(alpha=0.22)
            axi.set_xscale(xscale)
            axi.set_yscale(yscale)
            axi.set_title('Star Formation History')
            axi.set_xlabel('lookback time [Gyr]')
            axi.set_ylabel(r'SFR [$M_{Sol}/yr$]')
            axi.legend()
    fig.tight_layout()
    if save:
        plt.savefig(output_dir / filename, dpi=dpi)


def test1():
    a = 10.0
    return a 

# testing function
def test_plot_fig(a=None):
    if a is not None:
        a /= 1e9
    else:
        a = -10
    fig, ax = plt.subplots()
    ax.plot([0,1],[0,a])
    # plt.show()

def display_fits(theta_percentiles, ylabels=None):
    """
    Display function for median parameters and uncertainty from MCMC

    Parameters
    ----------
    theta_percentiles : ndarray of shape (3, nparameters)
        [16, 50, 84] percentiles of each parameter
    ylabels : list of str
        All parameter names
    """
    ndim = theta_percentiles.shape[1]
    median_theta = theta_percentiles[1]
    sigma_theta = np.diff(theta_percentiles, axis=0)

    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':   # type: ignore
            ipython_shell = True
    except:
        ipython_shell = False

    if ylabels is None:
        ylabels = [f'theta{i}' for i in range(ndim)]

    for i in range(ndim):
        if ipython_shell:
            ylabeli = ylabels[i].replace('_', r'\_')
            txt = fr"{ylabeli}={median_theta[i]:.3f}^{{+{sigma_theta[1][i]:.3f}}}_{{-{sigma_theta[0][i]:.3f}}}"
            display(Math(txt))
        else:
            print(f"{ylabels[i]}=\t{median_theta[i]:.3f}\t+{sigma_theta[1][i]:.3f}\t-{sigma_theta[0][i]:.3f}")


# ===========================================
# Useful functions
# ===========================================

def continuity_sfh_agebins_sfrs(zred, logsfr_ratios, logmass):
    """
    Calculate agebins, massbins, and SFRs from input parameters
    using input redshift, logsfr_ratios, logmass and tuniv from astropy.cosmology

    Parameters
    ----------
    zred : float
        input redshift
    logsfr_ratios : list or np.array
        continuity_sfh sfr parameters
    logmass : float
        input logmass

    Returns
    -------
    agebins : np.array of shape (nbins, 2)
        start and finish of each bin (unit in yrs)
    massbins : np.array of shape (nbins,)
        total formed mass in each bin
    sfrs : np.array of shape (nbins,)
        star forming rate in each bin
    """
    nbins = len(logsfr_ratios)+1
    tuniv = cosmology.age(zred).value
    tbinmax = 0.85 * tuniv * 1e9
    lim1, lim2 = 7.4772, 8.0
    agelims = ([0, lim1] +
               np.linspace(lim2, np.log10(tbinmax), nbins-2).tolist() +
               [np.log10(tuniv*1e9)])
    agebins = 10**(np.array([agelims[:-1], agelims[1:]]))
    agebins = agebins.T

    mass = 10**logmass
    sratios = 10**np.clip(logsfr_ratios, -10, 10)
    dt = agebins[:, 1] - agebins[:, 0]
    coeffs = np.array([ (1. / np.prod(sratios[:i])) * (np.prod(dt[1: i+1]) / np.prod(dt[: i]))
                        for i in range(nbins)])
    m1 = mass / coeffs.sum()
    massbins = m1 * coeffs
    sfrs = massbins / dt
    return agebins, massbins, sfrs

def delayed_tau_trunc(tl, tage, tau, ttrunc, sf_slope, scale):
    """
    Returns Star forming rate of prospector parametric SFH model
    Units of tl, tage, tau has to match (yr or Gyr)
    
    Parameters
    ----------
    tl : ndarray
        lookback time to evaluate SFR. 
        tl=0 is galaxy at its redshift; if tl>tage then SFR returns 0
    tage : float
        Total age of the galaxy
    tau : float
        e-folding time of the SFH
    ttrunc : float
        Truncation time of the SFH, counting from the start of galaxy
    sf_slope : float
        Truncation slope after ttrunc
    scale : float
        Overall scaling of the total SFR
    
    Returns
    -------
    sfr : ndarray
        Total SFR w.r.t input lookback time grid
    """
    tl_trunc = tage - ttrunc
    sfr1 = scale * (tage-tl) * np.exp(-(tage-tl)/tau) * np.where((tl>=tl_trunc) & (tl<=tage), 1, 0)
    sfr2 = scale * np.clip(ttrunc*np.exp(-ttrunc/tau) + sf_slope*(tl_trunc-tl), 0, None) * np.where((tl>=0) & (tl<=tl_trunc), 1, 0)
    sfr = sfr1 + sfr2
    return sfr


def parametric_sfrs(zred, 
                    tage_tuniv, 
                    logmass, 
                    tau, 
                    fburst, 
                    fage_burst, 
                    fage_trunc, 
                    sf_slope,
                    tl=None):
    """
    High level parametric SFR calculation, returns SFR with given prospector inputs, and relevant time in Gyrs
    
    Parameters
    ----------
    zred : float
        Galaxy redshift
    tage_tuniv : float
        Percentage of galaxy age with respect to tuniv at its redshift
    logmass : float
        Log10(M/Msol)
    tau : float
        e-folding time of its SFR, in unit of [Gyr]
    fburst : float
        Fraciton of mass that are from late burst
    fage_burst : float
        Percentage of galaxy burst time with respect to its tage
    fage_trunc : float
        Percentage of galaxy SFR truncation time w.r.t its tage
    sf_slope : float
        SFR truncation slope after fage_trunc
    tl : ndarray
        lookback time to evaluate SFR in unit [yrs]
        if not provided, automatically generated a 200 pt log-spaced grid
        tl=0 is galaxy at its redshift; if tl>tage then SFR returns 0
    
    Returns
    -------
    tage : float
        Total age calculated from astropy.Planck18.age, in [Gyr]
    tburst : float
        Burst time from when galaxy was borned, in [Gyr]
    ttrunc : float
        SFR truncation time from when galaxy was borned, in [Gyr]
    scale : float
        Scaling coefficient to make sure SFR returns correct formed mass
    tl : ndarray
        Returns lookback time grid that are used to evaluated SFR, in [Gyr]
    sfr : ndarray
        Star formation rate at each tl grid point
    """
    tuniv = cosmology.age(zred).value * 1e9
    tage = tuniv * tage_tuniv
    tburst = tage * fage_burst
    tau = tau*1e9
    mass = 10**logmass
    mass_noburst = mass * (1-fburst)
    ttrunc = tage * fage_trunc
    tage_after_trunc = tage - ttrunc
    area_delayed_tau = tau**2 * (1-(1+(ttrunc/tau))*np.exp(-ttrunc/tau))

    if ttrunc*np.exp(-ttrunc/tau) + sf_slope*tage_after_trunc > 0:
        area_trunc = ttrunc*np.exp(-ttrunc/tau)*tage_after_trunc + sf_slope*0.5*tage_after_trunc**2
    else:
        area_trunc = -0.5 * ttrunc**2/sf_slope * np.exp(-2*ttrunc/tau)
    scale = mass_noburst / (area_delayed_tau+area_trunc)

    if tl is None:
        tl = 10**(np.linspace(7, np.log10(tage), 200))
    sfr = delayed_tau_trunc(tl, tage, tau, ttrunc, sf_slope, scale)
    return tage/1e9, tburst/1e9, ttrunc/1e9, scale, tl/1e9, sfr


def continuity_sfh_percentiles_steps(flat_samples,
                                     theta=None,
                                     zred_idx=0,
                                     logmass_idx=2,
                                     logsfr_ratios_idx=np.array([3,4,5,6,7,8]),
                                     n_transition=20,
                                     transition_start_idx=3,
                                     percentiles=[16, 50, 84]):
    """
    From a given flat sample of all parameters and zred_idx, logsfr_ratios_idx, logmass_idx,
    calculate percentile SFRs in each bin, and append the last datapoint so that its ready for plt.step()

    Parameters
    ----------
    flat_samples : np.array of shape (nsamples, ndims)
        Flattened MCMC chain
    theta : prospector_mcmc.theta instance
        theta instance containing all theta metadata, will ignore _idx keywords
    zred_idx : int
        Where redshift is in the flat_sample MCMC parameters
    logmass_idx : int
        Where logmass is in the parameters
    logsfr_ratios_idx : np.array
        All indexes for logsfr_ratios in parameters
    n_transition : int
        How finely to chop during age lims where change in redshift cause agebins to change
        and calculate continuous percentiles within those regions
    transition_start_idx : int
        When does fine chopping starts for agebins, including lookback time 0 (starting agelim)
        First two agebins should always be fixed, hense default=3
    percentiles : list
        Percentiles to evaluate piecewise SFH
    
    Returns
    -------
    all_age_lims : np.array of shape (nsteps, )
        All the age limits, including finely chopped agelims between major bins (unit in yrs)
    qs_agebins_all_sfrs : np.array of shape (nsfh, nsteps)
        Evaluated percentile SFRs between all_age_lims, with last datapoint repeated for plt.step()
    """
    if type(percentiles) is int or type(percentiles) is float:
        percentiles = [percentiles]
    n_percentiles = len(percentiles)
    nbins = len(logsfr_ratios_idx)+1
    transition_idx = np.arange(transition_start_idx, nbins)
    # loop over each step in MCMC chain to calculate agebins and SFRs
    all_age_lims = np.zeros((flat_samples.shape[0], nbins+1)) #/1e9
    all_sfrs = np.zeros((flat_samples.shape[0], nbins))

    if theta is not None:
        zred_idx = theta.zred_idx
        logmass_idx = theta.logmass_idx
        logsfr_ratios_idx = theta.logsfr_ratios_idx

    for i, _ in enumerate(flat_samples):
        theta_i = flat_samples[i]
        zred_i = theta_i[zred_idx]
        logmass_i = theta_i[logmass_idx]
        logsfr_ratios_i = theta_i[logsfr_ratios_idx]
        agebins_i, _, sfrs_i = continuity_sfh_agebins_sfrs(zred_i, logsfr_ratios_i, logmass_i)

        age_lims_i = np.hstack([agebins_i[:,0], agebins_i[-1,1]]) #/1e9
        all_age_lims[i] = age_lims_i
        all_sfrs[i] = sfrs_i

    age_lims_tbins = np.zeros((n_transition)*len(transition_idx))

    qs_agebins_sfr = np.zeros((n_percentiles, nbins))
    qs_age_tbins_sfr = np.zeros((n_percentiles, (n_transition-1)*len(transition_idx)))

    # first calculate SFR percentiles in major bins
    qs_agebins_sfr = np.percentile(all_sfrs, percentiles, axis=0)

    # for each transitions, cut them into n_transition and 
    # loop over each subbins
    for i, bound_idxi in enumerate(transition_idx):
        bstart = np.min(all_age_lims[:,bound_idxi])
        bend = np.max(all_age_lims[:,bound_idxi])
        fine_steps = np.linspace(bstart, bend, num=n_transition)
        # save the actual age fine limites in age_lims_tbins
        age_lims_tbins[i*n_transition: (i+1)*n_transition] = fine_steps

        for j in range(n_transition-1):
            b0 = fine_steps[j]
            b1 = fine_steps[j+1]
            f0 = np.clip(all_age_lims[:,bound_idxi]-b0, 0, (b1-b0))
            f1 = np.clip(b1-all_age_lims[:,bound_idxi], 0, (b1-b0))
            # the average sfr for EACH chain within this subbin
            ave_sfrs_inbin = (f0 * all_sfrs[:,bound_idxi-1] + f1 * all_sfrs[:,bound_idxi]) / (f0+f1)
            qs_age_tbins_sfr[:, j+i*(n_transition-1)] = np.percentile(ave_sfrs_inbin, percentiles)

    # stitch age lims together, first few age lims are all the same, as well as the last limit
    age_all_lims = np.hstack([all_age_lims[0][0:np.min(transition_idx)], age_lims_tbins, all_age_lims[0][-1]])
    # create new sfrs array but the first few elements are the same as the original sfrs
    qs_agebins_all_sfrs = qs_agebins_sfr[:, :np.min(transition_idx)]
    # stitch percentile agebins and qs_age_tbins together, everytime n_transition points are added, also
    # stitch the next major bin to it
    for i, bound_idxi in enumerate(transition_idx):
        qs_agebins_all_sfrs = np.hstack([qs_agebins_all_sfrs, qs_age_tbins_sfr[:, i*(n_transition-1): (i+1)*(n_transition-1)], qs_agebins_sfr[:, bound_idxi][:,None]])

    # add the last element to sfr array so that they have the same dimension as agelims
    qs_agebins_all_sfrs = np.hstack([qs_agebins_all_sfrs, qs_agebins_all_sfrs[:,-1][:,None]])

    return age_all_lims, qs_agebins_all_sfrs

def parametric_sfh_percentiles(flat_samples,
                               theta=None,
                               zred_idx=0,
                               logmass_idx=2,
                               tage_tuniv_idx=3,
                               tau_idx=4,
                               fage_trunc_idx=5,
                               sf_slope_idx=6,
                               fburst_idx=7,
                               fage_burst_idx=8,
                               pts=200,
                               percentiles=[16, 50, 84]):
    """
    Returns selected percentiles of parametric SFH and lookback time grid in [Gyr]

    Parameters
    ----------
    flat_samples : np.array of shape (nsamples, ndims)
        Flattened MCMC chain
    theta : prospector_mcmc.theta instance
        theta instance containing all theta metadata, will ignore _idx keywords
    zred_idx : int
        Where redshift is in the flat_sample MCMC parameters
    logmass_idx : int
        Where logmass is in the parameters
    tage_tuniv_idx : int
        Where tage_tuniv is in parameters
    tau_idx : int
        Where tau is in parameters
    fage_trunc_idx : int
        Where fage_trunc is in parameters
    sf_slope_idx : int
        Where sf_slope is in parameters
    fburst_idx : int
        Where fburst is in parameters
    fage_burst_idx : int
        Where fage_burst is in parameters
    pts : int
        Number of time grid points to evaluate SFRs
    percentiles : list or float or int
        Percentiles to evaluate SFRs

    Returns
    -------
    common_tl : ndarray of shape (npts, )
        lookback time grid for all SFRs in [Gyr]
    sfh_percentiles : ndarray of shape (npercentiles, npts)
        SFR at all percentiles, w.r.t. common_tl
    tl_burst_percentiles : ndarray of shape (npercentiles, )
        Percentiles of burst lookback times
    """
    if theta is not None:
        zred_idx = theta.zred_idx
        logmass_idx = theta.logmass_idx
        tage_tuniv_idx = theta.tage_tuniv_idx
        tau_idx = theta.tau_idx
        fage_trunc_idx = theta.fage_trunc_idx
        sf_slope_idx = theta.sf_slope_idx
        fburst_idx = theta.fburst_idx
        fage_burst_idx = theta.fage_burst_idx

    nsamples = flat_samples.shape[0]
    all_sfrs = np.empty((nsamples, pts))
    all_tl_bursts = np.empty((nsamples))
    zreds = flat_samples[:,zred_idx]
    tage_tunivs = flat_samples[:,tage_tuniv_idx]
    logmasses = flat_samples[:, logmass_idx]
    taus = flat_samples[:, tau_idx] # * 1e9
    fbursts = flat_samples[:, fburst_idx]
    fage_bursts = flat_samples[:, fage_burst_idx]
    fage_truncs = flat_samples[:, fage_trunc_idx]
    sf_slopes = flat_samples[:, sf_slope_idx]

    tunivs = cosmology.age(zreds).value * 1e9
    tages = tunivs * tage_tunivs
    max_tage = np.max(tages)
    common_tl = 10**(np.linspace(7, np.log10(max_tage), pts))
    for i in range(nsamples):
        tagei, tbursti, _, _, _, sfri = parametric_sfrs(zreds[i],
                                                        tage_tunivs[i],
                                                        logmasses[i],
                                                        taus[i],
                                                        fbursts[i],
                                                        fage_bursts[i],
                                                        fage_truncs[i],
                                                        sf_slopes[i],
                                                        tl=common_tl)
        tl_burst = tagei - tbursti
        all_sfrs[i] = sfri
        all_tl_bursts[i] = tl_burst
    # loop over each common_tl pts to grab percentiles
    if type(percentiles) is float or type(percentiles) is int:
        percentiles = [percentiles]
    # n_percentiles = len(percentiles)
    sfh_percentiles = np.nanpercentile(all_sfrs, percentiles, axis=0)
    tl_burst_percentiles = np.nanpercentile(all_tl_bursts, percentiles)
    return common_tl/1e9, sfh_percentiles, tl_burst_percentiles

def get_theta_percentiles(samples, discard=1000, thin=10, percentiles=[16, 50, 84]):
    """
    Returns selected percentiles of thetas from input samples

    Parameters
    ----------
    samples : np.array
        MCMC chains, can be full sample or flattened
    discard : int
        if samples is the full chain, how many steps to discard
    thin : int
        if samples is the full chain, only sample every thin steps after discard
    percentiles : float or int or list
        Which percentile(s) to evaluate theta

    Returns
    -------
    theta_percentiles : np.array
        Evaluated percentile array of all theta
    """
    if type(percentiles) is int or type(percentiles) is float:
        percentiles = [percentiles]
    ndim = samples.shape[-1]
    if samples.ndim == 3:
        flat_samples = samples[discard+thin-1::thin, :, :].reshape(-1, ndim)
    else:
        flat_samples = samples

    theta_percentiles = np.percentile(flat_samples, percentiles, axis=0)
    # diff_percentiles = np.diff(percentiles, axis=0)
    return theta_percentiles



# ===========================================
# Main prospector_mcmc class
# ===========================================

class prospector_mcmc:
    """
    Flexible MCMC object that use custom prospector tools and emcee package.

    Parameters:
    -----------
    config : str, default=None
        YAML configuration file to read settings and parameters from
    sfh_stype : str
        SFH type used for this MCMC object, can be 'parametric_sfh' or 'continuity_sfh'
    nwalkers : int, default=32
        Number of walkers in the MCMC ensemble, can be changed
    jitter : float, default=1e-4
        Random shift sizes from initial point, can be changed
    nsteps : int, default=8000
        MCMC chain steps, can be changed
    discard : int, default=2000
        Number of steps to discard, can be changed
    thin : int, default=50
        However many samples to sample from the full chain, can be changed
    zprior : bool, default=True
        Whether to use input redshift and uncertainty as MCMC redshift gaussian prior
    filters : ndarray of shape (nfilt, 2, nsample) or str
        Filters from read_filters(), or str of the path to SPHEREx fiducial_filters.txt file
    parallel : bool, default=False
        Use python multiprocessing to speed up MCMC
    save_sampler : bool, default=False
        Whether to save the full sampler in a .h5 file
    sampler_filename : str, default='mcmc_results_sampler.h5'
        If save_sampler=True, the full mcmc sampler will be saved to this file
    
    Attributes
    ----------
    myparams : dict
        Our custom parameter dictionary format, either from config.yaml or created by default
    custom_model_obj : custom_prospector_tools.custom_prospector
        Our custom prospector object with filters as input
    initial_vals :
        All initial values for possible parameters
    theta :
        Actual free parameters that are going to be sampled. Contains:
        theta.dict : theta dictionary with initialized values
        theta.keys : key names for free parameters
        theta.n_each_theta : number of parameter within each parameters
        self.theta.initial : parameter initialized values
        self.theta.ndim : total number of parameters
        self.theta.zred_idx_in_theta : where zred is in theta (for updating prior)
        self.theta.logmass_idx_in_theta : where logmass is in theta (for converting to stellar mass)
    flat_mfracs : ndarray of shape (nsteps*nwalkers//thin, )
        Flattened surviving mass fraction from prospector along the chain, sampled with every "thin" steps

    Methods
    -------
    update_theta_infos :
        update theta settings depending on desired free parameters
    update_myparam_from_theta :
        create myparam from an input theta
    fit :
        Run MCMC with an input object's spectra and uncertainty
    get_results :
        Get all the key results from the finished MCMC chain
    """
    def __init__(self,
                 config=None,
                 sfh_type='',
                 nwalkers=None,
                 jitter=None,
                 nsteps=None,
                 discard=None,
                 thin=None,
                 zprior=None,
                 filters=None,
                 parallel=None,
                 save_sampler=None,
                 sampler_filename=None,
                 verbose=None
                 ):

        if config is not None:
            with open(config, 'r') as file:
                config = yaml.safe_load(file)
            spherex_id = config['Catalog']['SPHERExRefID']
            settings = config['MCMC_settings']
            self.nwalkers = nwalkers or settings['nwalkers']
            self.jitter = jitter or settings['jitter']
            self.nsteps = nsteps or settings['nsteps']
            self.discard = discard or settings['discard']
            self.thin = thin or settings['thin']

            if zprior is None:
                self.zprior = settings['zprior']
            else:
                self.zprior = zprior
            if parallel is None:
                self.parallel = settings['parallel']
            else:
                self.parallel = parallel
            if save_sampler is None:
                self.save_sampler = settings['save_sampler']
            else:
                self.save_sampler = save_sampler
            if sampler_filename is None:
                self.sampler_filename = settings['sampler_filename']
                if self.sampler_filename == 'use_id':
                    self.sampler_filename = f"mcmc_results_sampler_{spherex_id}.h5"
            else:
                self.sampler_filename = sampler_filename
            if verbose is None:
                self.verbose = settings['verbose']
            else:
                self.verbose = verbose

            if filters is None:
                filter_list = config['Catalog']['filter_list']
                self.filters = cpt.read_filters(filter_list)
            else:
                if type(filters) is str:
                    self.filters = cpt.read_filters(filters)
                else:
                    self.filters = filters

            self.myparams = config['Parameters']
            self.read_from_config = True

            # figure out from config parameters which SFH type it is
            try:
                self.myparams['nbins']
                self.sfh_type = 'continuity_sfh'
            except:
                self.sfh_type = 'parametric_sfh'
            self.custom_model_obj = cpt.custom_prospector(sfh_type=self.sfh_type, filters=self.filters)

        # if config.yaml is not provided
        else:
            if sfh_type == '':
                raise ValueError("sfh_type has to be given if config file is not provided!")
            elif (sfh_type != 'continuity_sfh') and (sfh_type != 'parametric_sfh'):
                raise ValueError("sfh_type has to be 'continuity_sfh' or 'parametric_sfh'")
            else:
                self.sfh_type = sfh_type
                if zprior is None:
                    zprior = True
                if parallel is None:
                    parallel = False
                if save_sampler is None:
                    save_sampler = False
                if sampler_filename is None:
                    sampler_filename = 'mcmc_results_sampler.h5'
                if verbose is None:
                    verbose = True
                self.nwalkers = nwalkers or global_defaults.nwalkers
                self.jitter = jitter or global_defaults.jitter
                self.nsteps = nsteps or global_defaults.nsteps
                self.discard = discard or global_defaults.discard
                self.thin = thin or global_defaults.thin
                self.zprior = zprior
                self.parallel = parallel
                self.read_from_config = False
                self.save_sampler = save_sampler
                self.sampler_filename = sampler_filename
                self.verbose = verbose

            # if config is not provided, user NEED to provide filter, so no checking if it's None
            if type(filters) is str:
                self.filters = cpt.read_filters(filters)
            else:
                self.filters = filters
            self.custom_model_obj = cpt.custom_prospector(sfh_type=self.sfh_type, filters=self.filters)
            # use the custom prospector object to create an initial template of myparams
            self.myparams = self.custom_model_obj.create_myparams()

        self.initial_vals = global_defaults.initial_vals(self.myparams)
        # set discard to ~20% of nstep if nstep < discard
        if self.nsteps <= self.discard:
            self.discard = int(self.nsteps // global_defaults.discard_fraction)

        # set up initial theta array from pre-defined values and theta_keys
        self.update_theta_infos(free_params='default', from_config=self.read_from_config)
        # this creates: self.theta.dict
                      # self.theta.keys
                      # self.theta.n_each_theta
                      # self.theta.initial
                      # self.theta.ndim
                      # self.theta.zred_idx
                      # self.theta.logmass_idx
                      # self.theta.logsfr_ratios_idx
                      # self.theta.ylabels

    def update_theta_infos(self, free_params=None, add_params=None, fix_params=None, from_config=False):
        """
        create or update theta_dict that keeps track of free parameters
        and fill in initial values for free parameters

        Create attributes below for actual free parameters that are going to be sampled:
            theta.dict : theta dictionary with initialized values
            theta.keys : key names for free parameters
            theta.n_each_theta : number of parameter within each parameters
            self.theta.initial : parameter initialized values
            self.theta.ndim : total number of parameters
            self.theta.zred_idx : where zred is in theta (for updating prior)
            self.theta.logmass_idx: where logmass is in theta (for converting to stellar mass)
            self.theta.logsfr_ratios_idx: indexes of all logsfr_ratios in theta (nparray)
            self.theta.ylabels: string labels for each parameter, useful for plotting

        Parameters
        ----------
        free_params : list, default=None
            Set free parameters to sample, ex: ['zred', 'logzsol', 'logmass']
            if free_params='default', use pre-defined default 15 free parameters
        add_params : list, default=None
            Add desired free parameters to existing free parameter lists
        fix_params : list, default=None
            Remove desired parameters from existing free parameter lists
        from_config : bool, default=False
            If True, create theta from config.yaml definitions
        """

        # create an empty theta dictionary and fill in keys based on config parameters
        # and set values to pre-defined initial values
        theta_dict = {}
        if from_config:
            for _, key in enumerate(list(self.myparams.keys())):
                if self.myparams[key] == 'Y':
                    theta_dict[key] = self.initial_vals[key]
        else:
            if free_params == 'default':
                # if free params are not specified, pick pre-defined default free parameters
                free_params = global_defaults.free_params()

            elif free_params is None:
                # if free_param is not provided, 
                # assume this is not the first time the function has been run
                # and self.theta already exist
                try:
                    free_params = self.theta.keys
                except:
                    raise NameError("self.theta.keys doesn't exist! make sure prospector_mcmc.update_theta_infos is run first with free_params='default'")

            # loop over myparams that has ever possible parameter, check if free_param has them, 
            # add ones that are listed in free_param to theta_dict
            for _, key in enumerate(list(self.myparams.keys())):
                if np.isin(key, free_params):
                    theta_dict[key] = self.initial_vals[key]
            # now force "add_param" inputs to be added to theta_dict
            # and remove any parameter listed in "fix_parms"
            if add_params is not None:
                for _, key in enumerate(add_params):
                    theta_dict[key] = self.initial_vals[key]
            if fix_params is not None:
                for _, key in enumerate(fix_params):
                    theta_dict.pop(key)
        # after theta_dict is constructed, grab all the key names
        theta_keys = list(theta_dict.keys())

        # loop over each theta parameter keys and count how many elements are in each parameter
        n_each_theta = np.zeros(len(theta_keys), dtype=int)
        for i, key in enumerate(theta_keys):
            n_each_theta[i] = len(np.atleast_1d(self.initial_vals[key]))
        # flatten the theta_dict to a 1-d array as theta_initial
        theta_initial = np.hstack(list(theta_dict.values()))
        n_theta = len(theta_initial)

        # figure out where zred and logmass is in theta
        zred_key_idx_in_theta_dict = np.argwhere(np.array(theta_keys)=='zred')[0][0]
        zred_idx_in_theta = np.sum(n_each_theta[:zred_key_idx_in_theta_dict])
        logmass_key_idx_in_theta_dict = np.argwhere(np.array(theta_keys)=='logmass')[0][0]
        logmass_idx_in_theta = np.sum(n_each_theta[:logmass_key_idx_in_theta_dict])

        # save all relevant theta related information as namespace below self.theta
        self.theta = SimpleNamespace()
        self.theta.dict = theta_dict
        self.theta.keys = theta_keys
        self.theta.n_each_theta = n_each_theta
        self.theta.initial = theta_initial
        self.theta.ndim = n_theta
        self.theta.zred_idx = zred_idx_in_theta
        self.theta.logmass_idx = logmass_idx_in_theta

        # create all label names
        self.theta.ylabels = []
        for i, key in enumerate(self.theta.keys):
            if key == 'logsfr_ratios':
                for j in range(self.theta.n_each_theta[i]):
                    self.theta.ylabels.append(f"logSFR_r{j+1}")
            else:
                self.theta.ylabels.append(key)

        # also figure out all the index numbers for logsfr_ratios
        if self.sfh_type == 'continuity_sfh':
            logsfr_ratios_key_idx_in_theta_dict = np.argwhere(np.array(theta_keys)=='logsfr_ratios')[0][0]
            logsfr_ratios_idx_in_theta_start = np.sum(n_each_theta[:logsfr_ratios_key_idx_in_theta_dict])
            self.theta.logsfr_ratios_idx = np.arange(logsfr_ratios_idx_in_theta_start, 
                                                     logsfr_ratios_idx_in_theta_start+n_each_theta[logsfr_ratios_key_idx_in_theta_dict])

        elif self.sfh_type == 'parametric_sfh':
            self.tage_tuniv_idx = np.argwhere(np.array(theta_keys)=='tage_tuniv')[0][0]
            self.tau_idx = np.argwhere(np.array(theta_keys)=='tau')[0][0]
            self.fburst_idx = np.argwhere(np.array(theta_keys)=='fburst')[0][0]
            self.fage_burst_idx = np.argwhere(np.array(theta_keys)=='fage_burst')[0][0]
            self.fage_trunc_idx = np.argwhere(np.array(theta_keys)=='fage_trunc')[0][0]
            self.sf_slope_idx = np.argwhere(np.array(theta_keys)=='sf_slope')[0][0]


    def update_myparam_from_theta(self, theta):
        """
        update myparam from each MCMC iteration's theta values for prospector to use
        """
        myparams = self.myparams.copy()
        iend = 0
        for i, key in enumerate(self.theta.keys):
            n_this_theta = self.theta.n_each_theta[i]
            istart = iend
            iend = istart + n_this_theta
            if n_this_theta > 1:
                myparams[key] = theta[istart:iend]
            else:
                myparams[key] = theta[istart:iend][0]
        return myparams

    def _log_likelihood(self, theta, flux, flux_error):
        myparams = self.update_myparam_from_theta(theta)
        _, _, model_flux, _ = self.custom_model_obj.generate_spectra(myparams=myparams)
        ll = -0.5 * np.sum((model_flux-flux)**2/flux_error**2 + np.log(2*np.pi*flux_error**2))
        return ll

    def _log_prior(self, theta, redshift=None, redshift_sigma=None):

        prior_funcs = global_defaults.default_priors(redshift=redshift, redshift_sigma=redshift_sigma)
        lps = []

        # loop over theta keys and add contributing prior, taken into account each parameters' length
        iend = 0
        for i, key in enumerate(self.theta.keys):
            n_this_theta = self.theta.n_each_theta[i]
            # loop over number of this parameters, for most parameter this is 1
            # for logsfr_ratios this is nbins-1
            for j in range(n_this_theta):
                istart = iend
                iend = istart + 1
                this_theta = theta[istart:iend][0]
                lps.append(prior_funcs[key](this_theta))

        lp = np.sum(lps)
        return lp
        
    def _log_probability(self, theta, flux, flux_error, redshift=None, redshift_sigma=None):
        lp = self._log_prior(theta, redshift=redshift, redshift_sigma=redshift_sigma)
        if not np.isfinite(lp):
            log_prob = lp
            mfrac = 0.0
        else:
            # log_prob = lp + self._log_likelihood(theta, flux, flux_error)
            myparams = self.update_myparam_from_theta(theta)
            _, _, model_flux, mfrac = self.custom_model_obj.generate_spectra(myparams=myparams)     # if filters is passed in here it is extremely slow
            ll = -0.5 * np.sum((model_flux-flux)**2/flux_error**2 + np.log(2*np.pi*flux_error**2))
            log_prob = lp + ll
        return log_prob, mfrac
    
    def run_mcmc(self, 
                 flux,
                 flux_error,
                 # flux_unit='uJy',
                 redshift=None,
                 redshift_sig=None,
                 initial=None,
                 nwalkers=None,
                 jitter=None,
                 nsteps=None,
                 zprior=None,
                 save_sampler=None,
                 sampler_filename=None,
                 progress=None,
                 parallel=None):
        """
        Run MCMC with prospector

        Parameters
        ----------
        flux : ndarray of shape (nf, )
            Input f_nu in uJy
        flux_error : ndarray of shape (nf, )
            Input flux flux_error in uJy
        redshift : float, default=None
            Input photoz as prior
        redshift_sig : float, default=None
            Input photoz uncertainty as gaussian prior std
        initial : ndarray of shape (ndim, )
            Initial position of all free parameters
        nwalkers : int, default=32
            Number of walkers in the MCMC ensemble, can be changed
        jitter : float, default=1e-4
            Random shift sizes from initial point, can be changed
        nsteps : int, default=8000
            MCMC chain steps, can be changed
        zprior : bool, default=True
            Whether to use input redshift and uncertainty as MCMC redshift gaussian prior
        save_sampler : bool, default=False
            Whether to save the full sampler in a .h5 file
        sampler_filename : str, default='mcmc_results_sampler.h5'
            If save_sampler=True, the full mcmc sampler will be saved to this file
        parallel : bool, default=False
            Use python multiprocessing to speed up MCMC
        progress : bool, default=True
            Whether to show Emcee progress bar or not
        """

        if initial is None:
            initial = self.theta.initial
        if redshift is not None:
            initial[self.theta.zred_idx] = redshift
        if not self.zprior:
            # change ONLY the argument redshift so that initial value is still input redshift, but not being used in prior
            redshift = None
            redshift_sig = None

        # check settings and use global if not provided
        nwalkers = nwalkers or self.nwalkers
        jitter = jitter or self.jitter
        nsteps = nsteps or self.nsteps
        # discard = discard or self.discard
        # thin = thin or self.thin
        zprior = zprior or self.zprior
        save_sampler = save_sampler or self.save_sampler
        sampler_filename = sampler_filename or self.sampler_filename
        parallel = parallel or self.parallel
        progress = progress or self.verbose

        # set up initial position for MCMC with jitters
        initial_pos = initial + jitter * np.random.randn(nwalkers, self.theta.ndim)
        
        if save_sampler:
            backend = emcee.backends.HDFBackend(sampler_filename)
            backend.reset(nwalkers, self.theta.ndim)
        else:
            backend = None
            # does it still need to be reset to prevent overwriting existing file?

        if parallel:
            init_args = (self.sfh_type, self.filters)
            with multiprocessing.Pool(processes=4, initializer=initialize_prospector, initargs=init_args) as pool:
                # --------------- use below for linux/windows -----------------
                # self.sampler = emcee.EnsembleSampler(nwalkers=nwalkers, 
                #                                 ndim=self.theta.ndim, 
                #                                 log_prob_fn=self._log_probability, 
                #                                 args=(flux, flux_error, redshift, redshift_sig),
                #                                 backend=backend,
                #                                 pool=pool)
                # --------------- use below for mac ----------------------
                self.sampler = emcee.EnsembleSampler(nwalkers=nwalkers, 
                                                ndim=self.theta.ndim, 
                                                log_prob_fn=global_log_probability, 
                                                args=(flux, 
                                                        flux_error, 
                                                        redshift, 
                                                        redshift_sig, 
                                                        self.sfh_type,
                                                        self.filters,
                                                        self._log_prior, 
                                                        self.update_myparam_from_theta),
                                                backend=backend,
                                                pool=pool)
                # -------------------------------------------------------
                self.sampler.run_mcmc(initial_pos, nsteps, progress=progress)

        else:
            self.sampler = emcee.EnsembleSampler(nwalkers=nwalkers, 
                                            ndim=self.theta.ndim, 
                                            log_prob_fn=self._log_probability, 
                                            args=(flux, flux_error, redshift, redshift_sig),
                                            backend=backend)
            self.sampler.run_mcmc(initial_pos, nsteps, progress=progress)

        # After MCMC is done
        self.full_samples = self.sampler.get_chain()

    def get_results(self, discard=None, thin=None, wl_min=0.2, wl_max=25):
        """
        return flat_samples, flat_mfracs, theta_percentiles, med_myparam, med_lbs, med_spectra, med_flux_conv, med_mfrac

        Parameters
        ----------
        discard : int
            How many samples in the chain to discard
        thin : int
            How many steps to skip along the chain to thin it
        wl_min : float, default=0.2
            lower bound of wavelength for prospector to output rest-frame spectra in [um]
        wl_max : float, default=25
            upper bound of wavelength for prospector to output rest-frame spectra in [um]

        Returns
        -------
        results : dict
            Compiled results from the flat chain with many useful quantities
        """
        discard = discard or self.discard
        thin = thin or self.thin

        nsteps = self.full_samples.shape[0]
        if discard >= nsteps:
            raise ValueError('discard cannot be equal or larger than nsteps!')

        flat_samples = self.sampler.get_chain(discard=discard, thin=thin, flat=True)
        flat_mfracs = self.sampler.get_blobs(discard=discard, thin=thin, flat=True)
        theta_percentiles = get_theta_percentiles(flat_samples, percentiles=[16, 50, 84])
        mfracs_percentiles = np.percentile(flat_mfracs, [16, 50, 84])
        med_myparam = self.update_myparam_from_theta(theta_percentiles[1])
        med_lbs, med_spectra, med_flux_conv, med_mfrac = self.custom_model_obj.generate_spectra(med_myparam, wl_min=wl_min, wl_max=wl_max)

        autocorr = self.sampler.get_autocorr_time(discard=discard, thin=thin, tol=0)

        # if 'logsfr_ratios' in self.theta.ylabels:
        #     sfh_type = 'continuity_sfh'
        # else:
        #     sfh_type = 'parametric_sfh'

        if self.sfh_type == 'continuity_sfh':
            med_agebins, med_massbins, med_sfrs = continuity_sfh_agebins_sfrs(med_myparam['zred'],
                                                                              med_myparam['logsfr_ratios'],
                                                                              med_myparam['logmass'])
            results = {
                'flat_samples': flat_samples,
                'flat_mfracs': flat_mfracs,
                'theta_percentiles': theta_percentiles,
                'med_myparam': med_myparam,
                'med_lbs': med_lbs,
                'med_spectra': med_spectra,
                'med_flux_conv': med_flux_conv,
                'med_mfrac': med_mfrac,
                'mfracs_percentiles': mfracs_percentiles,
                'med_agebins': med_agebins,
                'med_massbins': med_massbins,
                'med_sfrs': med_sfrs,
                'ylabels': self.theta.ylabels,
                'autocorr': autocorr
            }
        elif self.sfh_type == 'parametric_sfh':
            tage, tburst, ttrunc, scale, tl, sfr = parametric_sfrs(med_myparam['zred'],
                                                                   med_myparam['tage_tuniv'],
                                                                   med_myparam['logmass'],
                                                                   med_myparam['tau'],
                                                                   med_myparam['fburst'],
                                                                   med_myparam['fage_burst'],
                                                                   med_myparam['fage_trunc'],
                                                                   med_myparam['sf_slope'])
            results = {
                'flat_samples': flat_samples,
                'flat_mfracs': flat_mfracs,
                'theta_percentiles': theta_percentiles,
                'med_myparam': med_myparam,
                'med_lbs': med_lbs,
                'med_spectra': med_spectra,
                'med_flux_conv': med_flux_conv,
                'med_mfrac': med_mfrac,
                'mfracs_percentiles': mfracs_percentiles,
                'med_tage': tage,
                # 'med_tau': med_myparam['tau'],
                'med_tburst': tburst,
                'med_ttrunc': ttrunc,
                'med_sfh_scale': scale,
                'med_lookback_time': tl,
                'med_sfrs': sfr,
                # 'med_sf_slope': med_myparam['sf_slope']
                'ylabels': self.theta.ylabels,
                'autocorr': autocorr
            }

        return results
        # return flat_samples, flat_mfracs, theta_percentiles, med_myparam, med_lbs, med_spectra, med_flux_conv, med_mfrac


    # TODO change initial and prior from config or after initializing mcmc class
    # TODO modify custom_prospector to eventually use zcontinuous=2



# ===========================================
# Main function for the script
# ===========================================

def main():

    start_datetime = datetime.now().isoformat(timespec='seconds')
    args = parse_args()
    config = args.config

    # read config files first
    with open(config, 'r') as file:
        config_content = yaml.safe_load(file)
    spherex_id = config_content['Catalog']['SPHERExRefID']
    filename = config_content['Catalog']['filename']
    filter_list = config_content['Catalog']['filter_list']

    save_plots = config_content['MCMC_settings']['save_plots']
    plots_dir = config_content['MCMC_settings']['plots_dir']
    output_dir = config_content['MCMC_settings']['output_dir']
    output_filename = config_content['MCMC_settings']['output_filename']

    
    # is argparse keywords are provided when running script, override the settings from config
    if args.spherex_id is not None:
        spherex_id = args.spherex_id
    if args.filename is not None:
        filename = args.filename
    if args.no_plots is not None:
        save_plots = not args.no_plots
    if args.plots_dir is not None:
        plots_dir = args.plots_dir
    if args.output_dir is not None:
        output_dir = args.output_dir
    if args.output_filename is not None:
        output_filename = args.output_filename
    if args.filter_list is not None:
        filter_list = args.filter_list

    try:
        filter_list_filename = filter_list.split('/')[-1]
        filter_central_wavelengths = filter_list.replace(filter_list_filename, 'fiducial_filters_cent_waves.txt')
        lamb_obs = np.genfromtxt(filter_central_wavelengths, delimiter=' ')[:,1]
    except:
        filters = cpt.read_filters(filter_list)
        nfilt = filters.shape[0]
        lamb_obs = np.zeros(nfilt)
        threshold = 0.1
        for i in range(nfilt):
            lamb_i = filters[i][0]
            res_i = filters[i][1] / np.max(filters[i][1])
            mask = res_i > threshold
            lamb_obs[i] = np.sum(lamb_i[mask]*res_i[mask])/np.sum(res_i[mask])

    # the following settings can be passed to prospector_mcmc as None, in which case config will be used
    # if any of them is not None, override the config
    nwalkers = args.nwalkers                    # None or int
    jitter = args.jitter                        # None or int
    nsteps = args.nsteps                        # None or int
    discard = args.discard                      # None or int
    thin = args.thin                            # None or int
    save_sampler = args.save_sampler            # None or True
    sampler_filename = args.sampler_filename    # None or str
    parallel = args.parallel                    # None or True

    # Following settings are also passed to prospector_mcmc class, but since they are opposite of argparse, 
    # need to explicitly check their values, and pass None if not provided
    if args.no_zprior is not None:
        zprior = not args.no_zprior                 # Not (True) -> False
    else:
        zprior = None
    if args.quiet is not None:
        verbose = not args.quiet                    # Not (True) -> False
    else:
        verbose = True

    if sampler_filename == 'use_id':
        sampler_filename = f"mcmc_results_sampler_{spherex_id}.h5"
    if output_filename == 'use_id':
        output_filename = f"mcmc_results_report_{spherex_id}.h5"

    if verbose:
        print('Read input catalog...')
    # read the parquet catalog file
    cat = catalog_dataset(filename=filename)
    cat.get_row(SPHERExRefID=spherex_id)
    zspec = cat.zspec
    zphot = cat.zphot
    zphot_u68 = cat.zphot_u68
    zphot_l68 = cat.zphot_l68
    zphot_std = cat.zphot_std
    zsig = (zphot_u68-zphot_l68)/2
    spec = cat.spec
    err = cat.err
    frac102 = cat.frac102
    nonzeros = err != 50000.0
    external_phots = cat.external_phots

    # cat = catalog(filename=filename)
    # idx = np.where(cat.spherex_ids == spherex_id)[0][0]
    # zspec = cat.zspecs[idx]
    # zphot = cat.zphots[idx]
    # zphot_u68 = cat.zphots_u68[idx]
    # zphot_l68 = cat.zphots_l68[idx]
    # zphot_std = cat.zphots_std[idx]
    # zsig = (zphot_u68-zphot_l68)/2
    # # zsig = cat.zsigs[idx]
    # spec = cat.spectra[idx]
    # err = cat.error[idx]
    # frac102 = cat.frac102[idx]
    # nonzeros = err != 50000.0
    # external_phots = cat.get_external_phots(SPHERExRefID=spherex_id)

    if verbose:
        print('Creating MCMC instance...')
    tstart = time.time()
    pmcmc = prospector_mcmc(config=config,
                            nwalkers=nwalkers,
                            jitter=jitter,
                            nsteps=nsteps,
                            discard=discard,
                            thin=thin,
                            zprior=zprior,
                            filters=filter_list,
                            parallel=parallel,
                            save_sampler=save_sampler,
                            sampler_filename=sampler_filename,
                            verbose=verbose
                            )
    
    if verbose:
        print(f'SPHERExRefID:\t{spherex_id}')
        print(f'nwalkers:\t{pmcmc.nwalkers} \
            \n\tjitter:  \t{pmcmc.jitter} \
            \n\tnsteps:  \t{pmcmc.nsteps} \
            \n\tdiscard: \t{pmcmc.discard} \
            \n\tthin:    \t{pmcmc.thin} \
            \n\tzprior:  \t{pmcmc.zprior} \
            \n\tparallel:\t{pmcmc.parallel}')
        
        print('Run MCMC...')
    # run MCMC
    pmcmc.run_mcmc(flux=spec, flux_error=err, redshift=zphot, redshift_sig=zsig)

    # extract results
    mcmc_results = pmcmc.get_results()
    med_myparam = mcmc_results['med_myparam']
    theta_percentiles = mcmc_results['theta_percentiles']

    tend = time.time()

    if verbose:
        display_fits(theta_percentiles=theta_percentiles, ylabels=pmcmc.theta.ylabels)


    # ------------- save plot block ---------------
    if save_plots:
        plot_chain(pmcmc.full_samples,
                   ylabels=pmcmc.theta.ylabels,
                   save=True,
                   filename=f'mcmc_results_chains_{spherex_id}.png',
                   output_dirname=plots_dir)

        plot_corner(mcmc_results['flat_samples'],
                    ylabels=pmcmc.theta.ylabels,
                    save=True,
                    filename=f'mcmc_results_corner_{spherex_id}.png',
                    output_dirname=plots_dir
                    )

        # get model and SFH percentiles for sed_sfh plot
        if pmcmc.sfh_type == 'continuity_sfh':
            qs_agelims, qs_agebins_all_sfrs = continuity_sfh_percentiles_steps(
                mcmc_results['flat_samples'],
                theta=pmcmc.theta,
                n_transition=50,
                transition_start_idx=3,
                percentiles=[16,50,84]
            )

            plot_sed_sfh(lamb_obs,
                         spec,
                         err,
                         lamb_model=mcmc_results['med_lbs']*(1+theta_percentiles[1, pmcmc.theta.zred_idx]),
                         spec_model=mcmc_results['med_spectra'],
                         agelims_model=mcmc_results['med_agebins'],
                         sfrsteps_model=mcmc_results['med_sfrs'],
                         qs_agelims=qs_agelims,
                         qs_sfrsteps=qs_agebins_all_sfrs,
                         external_phots=external_phots,
                         save=True,
                         filename=f"mcmc_results_sed_sfh_{spherex_id}.png",
                         output_dirname=plots_dir,
                         title_kwargs={
                            'spherex_id': spherex_id,
                            'zspec': zspec,
                            'zphot': zphot,
                            'zphot_u68': zphot_u68,
                            'zphot_l68': zphot_l68,
                            'zmcmc_med': theta_percentiles[1, pmcmc.theta.zred_idx],
                            'zmcmc_16': theta_percentiles[0, pmcmc.theta.zred_idx],
                            'zmcmc_84': theta_percentiles[2, pmcmc.theta.zred_idx],
                            'frac102': frac102,
                            'fontsize': 9,
                            }
                         )

        elif pmcmc.sfh_type == 'parametric_sfh':
            qs_tl, qs_sfrs, qs_tl_burst = parametric_sfh_percentiles(
                mcmc_results['flat_samples'],
                theta=pmcmc.theta,
                pts=200,
                percentiles=[16, 50, 84]
            )

            plot_sed_sfh(lamb_obs,
                         spec,
                         err,
                         lamb_model=mcmc_results['med_lbs']*(1+med_myparam['med_zred']),
                         spec_model=mcmc_results['med_spectra'],
                         tl_model=mcmc_results['med_lookback_time'],
                         sfr_model=mcmc_results['med_sfrs'],
                         tl_burst_model=mcmc_results['med_tage']-mcmc_results['med_tburst'],
                         qs_tl=qs_tl,
                         qs_sfrs=qs_sfrs,
                         qs_tl_burst=qs_tl_burst,
                         external_phots=external_phots,
                         save=True,
                         filename=f"mcmc_results_sed_sfh_{spherex_id}.png",
                         output_dirname=plots_dir,
                         title_kwargs={
                            'spherex_id': spherex_id,
                            'zspec': zspec,
                            'zphot': zphot,
                            'zphot_u68': zphot_u68,
                            'zphot_l68': zphot_l68,
                            'zmcmc_med': theta_percentiles[1, pmcmc.theta.zred_idx],
                            'zmcmc_16': theta_percentiles[0, pmcmc.theta.zred_idx],
                            'zmcmc_84': theta_percentiles[2, pmcmc.theta.zred_idx],
                            'frac102': frac102,
                            'fontsize': 9,
                            }
                         )
    # ------------- save plot block ---------------


    elapsed_time = tend - tstart
    end_datetime = datetime.now().isoformat(timespec='seconds')

    # output_dir = Path(output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)

    compressed_cols = ['flat_samples', 'flat_mfracs', 'med_lbs', 'med_spectra']

    # saving data to .h5 file
    save_h5_results(mcmc_results, 
                    output_filename=output_filename, 
                    output_dir=output_dir,
                    metadata={
                            'ylabels': mcmc_results['ylabels'],
                            'start_time': start_datetime,
                            'end_time': end_datetime,
                            'SPHERExRefID': spherex_id,
                            'zspec': zspec,
                            'zphot': zphot,
                            'zphot_u68': zphot_u68,
                            'zphot_l68': zphot_l68,
                            'zphot_std': zphot_std,
                            'nwalkers': pmcmc.nwalkers,
                            'jitter': pmcmc.jitter,
                            'nsteps': pmcmc.nsteps,
                            'discard': pmcmc.discard,
                            'thin': pmcmc.thin,
                            'zprior': pmcmc.zprior,
                            'parallel': pmcmc.parallel
                        }
                    )

    print(f'MCMC runtime = {elapsed_time} seconds')

    # # mcmc_results.pop('flat_samples')
    # np.savez_compressed(Path(output_dir) / f'{output_filename}.npz', mcmc_results=mcmc_results)

    # output_reports = {
    #     'Time': {
    #             'Start': start_datetime,
    #             'End': end_datetime
    #             },
    #     'Catalog': {
    #         'SPHERExRefID': spherex_id,
    #         'zspec': float(zspec),
    #         'zphot': float(zphot),
    #         'zphot_u68': float(zphot_u68),
    #         'zphot_l68': float(zphot_l68),
    #         },
    #     'MCMC_settings': {
    #         'nwalkers': pmcmc.nwalkers,
    #         'jitter': float(pmcmc.jitter),
    #         'nsteps': pmcmc.nsteps,
    #         'discard': pmcmc.discard,
    #         'thin': pmcmc.thin,
    #         'zprior': pmcmc.zprior,
    #         'parallel': pmcmc.parallel
    #         },
    #     }
    # param_reports = {}
    # for i, key in enumerate(pmcmc.theta.ylabels):
    #     param_reports[key] = mcmc_results['theta_percentiles'][:,i].tolist()

    # output_reports['MCMC_results'] = param_reports

    # with open(Path(output_dir) / f'{output_filename}.yaml', 'w') as output_file:
    #     yaml.safe_dump(output_reports, output_file, sort_keys=False)

if __name__ == '__main__':
    main()

