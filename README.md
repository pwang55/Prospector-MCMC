# Introduction

This repo contains several scripts and tools for using prospector and running MCMC with prospector.  

Some notable packages required for this repo:
- prospector (and FSPS, python-fsps)
- sedpy
- h5py
- numba
- astropy
- emcee
- corner.py

# Usages

To run MCMC on a selected SPHEREx galaxy,  
```
$ python prospector_mcmc.py --config mcmc_config.yaml
```

`--config` is required keyword to run in CLI mode.  
Modifying `mcmc_config.yaml` for custom options.  

For command line interface, There are many optional keywords. They will override the config file. Possible options are:  
- --spherex_id (-si): Override config SPHERExRefID for MCMC fits
- --filename (-f): override input L4 catalog
- --filter_list (-fl): SPHEREx fiducial_filters.txt path
- --nwalkers (-nw): number of emcee ensemble walkers
- --jitter (-j): random jitter from MCMC initial position
- --nsteps (-ns): number of emcee chain steps
- --discard (-d): number of emcee steps to discard for the output results. If discard>nstep, discard will be set to 1/5 of nsteps
- --thin (-t): number of steps to skip to think the chains
- --no_zprior (-nzp): if set, MCMC will not use zphot and error as gaussian prior (but will still use it as initial)
- --parallel (-p): Use python multiprocessing (currently weird and not recommended)
- --save_sampler (-ss): Save sampler to a large .h5 file
- --sampler_filename (-sf): If --save_sampler, save sampler backend to this filename (if "use_id", filename will be "mcmc_results_sampler_[spherex_id].h5")
- --output_filename (-o): Output report and results file name (if "use_id", filename will be "mcmc_results_[spherex_id].h5")
- --no_plots (-np): Don't save plots to plots_dir
- --plots_dir (-pd): Output plots directory
- --quiet (-q): Silence the verbose outputs

`prospector_mcmc.py` can also be imported in jupyter notebook for interactive runs via
```
import prospector_mcmc as pm_lib
pmcmc = pm_lib.prospector_mcmc(config='mcmc_config.yaml')
pmcmc.run_mcmc()
```

# Modules

## `custom_prospector_tools.py`:  
This script contains useful functions and a custom prospector class for quickly generating galaxy sed from a set of parameters.  

> ### read_filters(filter_list, half_length=105)  
>  ---
> 
> Read SPHEREx filters and return ndarray in shape (nfilt, 2, 2*half_length)  
> This function ensures filter convolutions are quick by making sure they are all the same size.  
> `filter_list` should be the path to the file `fiducial_filters.txt`.  
> If `fiducial_filters_cent_waves.txt` is in the same folder, central wavelengths will be read from it; otherwise rough estimates will be calculated from filters themselves.
>  
> ### f_convolve_filter(wl, flux)
> ---
> Convolve spectra with filters read from read_filters().  
> It uses np.trapezoid which itself has some approximation.  
> 
> ### custom_prospector(sfh_type)
> ---
> This is the custom prospector class that is currently used by prospector_mcmc.py.  


## `prospector_mcmc.py`
This script is the main mcmc code that can be run in command line, or imported to construct class in python notebooks.  

Some notable functions and class are:  
> ### default_settings()
> ---
> Currently, all default settings for initial values, priors and default model parameters are hard-coded here. It will likely be changed in the future, but for now, this is > the place if we want to change any default behavior.  
> ### catalog(filename)
> ---
> This is the class that handles reading SPHEREx L4 catalog and create some useful arrays such as zspecs, spectra, etc. It also has a funciton `get_external_phots` that grabs > refcat photometry for a given SPHERExRefID (or row index of this file).  
> ### read_h5_results(h5file)
> ---
> This is a useful function to read back the output results from `prospector_mcmc.py`, as for now the script generates `.h5` files that are easy to handle metadata and different data format, but not so easy to read. This function returns a python dictionary.  

### Plotting and display functions:
> | Functions | Description |
> | :--- | :--- |
> | plot_chain() | Plot full MCMC chains  |
> | plot_corner() | Plot corner plot from MCMC flattened results  |
> | plot_sed_sfh() | Plot median fit SED and SFH |
> | display_fits() | Display parameter fits on screen |

### Useful functions for conversions, calculations, etc:
> | Functions | Description |
> | :--- | :--- |
> | continuity_sfh_agebins_sfrs() | Calculate agebins, massbins, and sfrs from redshift, logsfr_ratios, logmass. |
> | continuity_sfh_percentiles_steps() | Input the flat_samples from MCMC results, returns the age limits and percentile SFRs in the format that is ready for plt.step() |
> | get_theta_percentiles() |  Given a full MCMC sample chain or flattened sample, return the requested percentiles of each parameter. |
> | prospector_mcmc | Main prospector mcmc class that can be constructed after importing this script.  |
