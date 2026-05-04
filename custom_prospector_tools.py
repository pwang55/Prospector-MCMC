import os
# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from sedpy.observate import Filter, load_filters
from pathlib import Path
from prospect.models.templates import TemplateLibrary, describe, adjust_continuity_agebins
import prospect.models.transforms as transforms
from prospect.models import SpecModel
from prospect.sources import CSPSpecBasis, FastStepBasis
from astropy.cosmology import Planck18
import yaml
import emcee
import os
import multiprocessing
from scipy.stats import norm, t, lognorm, loguniform
from functools import partial
from numba import jit, njit

cosmology = Planck18

# Read SPHEREx filters
def read_filters(filter_list, half_length=105, return_lamb_obs=False, response_threshold=0.1):
    '''
    Read SPHEREx official filters and return ndarray of (nfilt, 2, 2*half_length)
    
    Parameters
    ----------
    filter_list : str
        Full path to the filter_list.txt that contains all the filter names in order
    half_length : int, default=105
        Half size of each filter's length.
        Currently, official SPHEREx filters don't have the same size; half_length of 105 is a good middle ground for 306 bands
        to cut off unimportant outside parts while preserving response to <3% accuracy.
        If half_length*2 > existing wavelength grid, fill longer wavelength side with zeros
    return_lamb_obs : bool, default=False
        If True, also returns the lamb_obs by finding 'fiducial_filters_cent_waves.txt' at the same directory as the filter_list
        If the cent_wave file can't be found, calculate by weight average of normalized response>0.1
        
    Returns
    -------
    filters : ndarray of shape (nfilt, 2, 2*half_length)
        Full array of all the filters' wavelength coverage (micron) and response
        For example, n_filter=i has wavelength filters[i][0] and response filters[i][1]
    lamb_obs : ndarray of shape (nfilt, )
        Only return when return_lamb_obs=True
    '''
    filter_dir = Path(filter_list).parent
    filter_names = np.loadtxt(filter_list, dtype=str)
    Nf = filter_names.shape[0]
    filters = np.zeros((Nf, 2, half_length*2))

    for i in range(Nf):
        filter_path_name = filter_dir / Path(filter_names[i])
        filt_i = np.loadtxt(filter_path_name)
        wavelength_i = filt_i[:,0] * 1e-4   # convert from AA to micron
        response_i = filt_i[:,1]
        arg_peak = np.argmax(response_i)
        if len(wavelength_i) < half_length*2:
            wavelength_i1 = np.zeros(half_length*2)
            response_i1 = np.zeros(half_length*2)
            wavelength_i1[:len(wavelength_i)] = wavelength_i
            dw = wavelength_i[-1] - wavelength_i[-2]
            wavelength_i1[len(wavelength_i):] = wavelength_i[-1] + dw * np.arange(1, len(wavelength_i1)-len(wavelength_i)+1)
            response_i1[:len(response_i)] = response_i
        else:
            if arg_peak - half_length < 0:
                istart = 0
                ifinish = half_length*2
            elif arg_peak + half_length > len(wavelength_i):
                ifinish = len(wavelength_i)
                istart = ifinish - half_length*2
            else:
                istart = arg_peak - half_length
                ifinish = arg_peak + half_length
            wavelength_i1 = wavelength_i[istart:ifinish]
            response_i1 = response_i[istart:ifinish]
        tot_response_i1 = np.trapezoid(response_i1, wavelength_i1)
        response_i1 = response_i1 / tot_response_i1    # divide by total response now so that when convolving SED with filters no normalization is needed
        filters[i][0] = wavelength_i1
        filters[i][1] = response_i1
        # filters.append((wavelength_i1, response_i1))
    if return_lamb_obs:
        try:
            filter_central_wavelengths = filter_list.replace('fiducial_filters.txt', 'fiducial_filters_cent_waves.txt')
            lamb_obs = np.genfromtxt(filter_central_wavelengths)[:,1]
        except:
            lamb_obs = np.zeros(Nf)
            for i in range(Nf):
                wav_i = filters[i][0]
                res_i = filters[i][1] / np.max(filters[i][1])
                mask = res_i > response_threshold
                lamb_obs[i] = np.sum(wav_i[mask] * res_i[mask]) / np.sum(res_i[mask])
        return filters, lamb_obs
    else:
        return filters
    
@njit(fastmath=True)
def f_convolve_filter(wl, flux, filters=None):
    '''
    Convolve spectra with filters read from read_filters()
    filters need to have shape (nfilt, 2, ns)

    Parameters
    ----------
    wl : ndarray of shape n
        wavelength points in micron
    flux : ndarray of shape n
        flux in f_nu
    filters : ndarray of shape (nfilt, 2, ns)
        filters array from read_filters()

    Returns
    -------
    flux_conv : ndarray of shape (nfilt, )
        Convolved fluxes in each passband
    '''
    if filters is not None:
        Nf = len(filters)
        flux_conv = np.zeros(Nf)
        for i in range(Nf):
            lb = filters[i][0]
            ftrans = filters[i][1]
            f_interp = np.interp(lb, wl, flux)
            fnu_i = np.trapezoid(f_interp*ftrans, lb)
            flux_conv[i] = fnu_i
    else:
        flux_conv = flux
    return flux_conv

class custom_prospector:
    """
    Our custom class for generating prospector spectra

    Parameters
    ----------
    sfh_type : str, default='continuity_sfh'
        Can be 'parametric_sfh' or 'continuity_sfh'
    filters : numpy.ndarray of shape (nfilt, 2, nsample) or list
        numpy.ndarray of shape (nfilt, 2, nsample) from read_filter(), 
        or a list of sedpy filter objects

    Attributes
    ----------
    myparams_example : dict
        Example template of myparams for user to modify, our parameter dictionary format
        this will be translate to prospector model_param
    model_params : dict
        Prospector readable parameter dictionar, will be generated based on myparam
        User should interact with myparams to make sure parameters all make sense
    obs : dict
        Prospector observation dictionary, contains filters if given
    sfh_type : str
        Same as input sfh_type
    wavelength : numpy.ndarray of shape (nl, )
        Output spectra wavelength grid, create after running custom_prospector.generate_spectra
    spectra : numpy.ndarray of shape (nl, )
        Output spectra f_nu, create after running custom_prospector.generate_spectra
        Default output unit is uJy
    flux_conv : numpy.ndarray of shape (nf, )
        Output convolved fluxes of each bandpass, create after running custom_prospector.generate_spectra
        Will be zero if filters not provided
    mfrac : float
        Surviving mass fraction from SPS model, create after running custom_prospector.generate_spectra
        In unit of solar mass

    Methods
    -------
    creat_myparams() :
        Create a convenience custom parameter dictionary for our user to interface with
    convert_myparams_model_params(params) : 
        Convert a given myparam format into Prospector model_param format,
        and making necessary calculations such as mass in each bins, agebins, etc
    generate_spectra(myparams=None, model_params=None, wl_min=0.2, wl_max=6, filters=None, unit='uJy') :
        Using a given myparams or a model_params to create spectra using Prospector

    """
    def __init__(self, sfh_type='continuity_sfh', filters=None, zcontinuous=2, pmetal=-1.0):
        self.sfh_type = sfh_type

        # some pre-fixed parameters
        self.imf_type = 1
        self.dust_type = 4
        
        # depending on sfh_type, create the basis sps object from fsps
        if self.sfh_type == 'parametric_sfh':
            self.sps = CSPSpecBasis(zcontinuous=zcontinuous, pmetal=pmetal)
            self.sfh_init = 5
        elif self.sfh_type == 'continuity_sfh':
            self.sps = FastStepBasis(zcontinuous=zcontinuous, pmetal=pmetal)
            self.sfh_init = 3

        self.model_params = TemplateLibrary[sfh_type]
        self.model_params.update(TemplateLibrary['dust_emission'])
        self.model_params.update(TemplateLibrary['nebular'])
        self.model_params.update(TemplateLibrary['agn'])

        # create the convenience custom parameter dictionary
        self.myparams_example = self.create_myparams()

        self.filters = filters
        self.obs = {}
        if filters is not None:
            if type(filters) == np.ndarray:
                filterslist_sedpy = []
                filters_A = filters.copy()
                filters_A[:,0,:] *= 1e4
                filters_A[:,1,:][filters_A[:,1,:]<1e-5] = 1e-5
                Nfilt = filters_A.shape[0]
                filterslist_sedpy = []
                # np.trapz = np.trapezoid # sedpy annoyingly use np.trapz, which has been removed from newer numpy
                for i in range(Nfilt):
                    filterslist_sedpy.append(Filter(data=filters_A[i], kname=f'spherex_{str(i+1).zfill(3)}', min_trans=1e-6))                
                self.obs['filters'] = filterslist_sedpy
            elif type(filters) == list:
                self.obs['filters'] = filters

    def create_myparams(self):
        """
        Create a convenience custom parameter dictionary for our user to interface with

        Returns
        -------
        myparams : dict
            Pre-defined default myparams dictionary
        """

        myparams = {
            'zred': 0.1,
            'logzsol': 0.0,
            'logmass': 10
        }
        if self.sfh_type == 'continuity_sfh':
            # tuniv = cosmology.age(z)
            myparams['nbins'] = 7
            # add logsfr_ratio keyword that use nbins-1 as length
            myparams['logsfr_ratios'] = [0.0] * (myparams['nbins']-1)


        elif self.sfh_type == 'parametric_sfh':
            parametric_sfh_params = {
                'tage_tuniv': 0.9,  # fraction of tuniv
                'tau': 1,   # Gyr
                'fage_trunc': 0.9,  # fraction of tage
                'sf_slope': -1.0,
                'fburst': 0.0,
                'fage_burst': 0.8   # fraction of tage
            }
            myparams.update(parametric_sfh_params)

        # dust attenuation
        dust_att_params = {
            # 'dust1': 0.0,
            'dust2': 0.0,
            'dust_ratio': 0.0,
            # 'dust1_index': -1.0,
            'dust_index': -1.0,
        }
        myparams.update(dust_att_params)
        # dust emission
        duste_params = {
            'duste_gamma': 0.01,
            # 'duste_loggamma': -2,
            'duste_umin': 0.01,
            'duste_qpah': 0.0
        }
        myparams.update(duste_params)
        # nebular emission
        nebular_params = {
            'add_neb_emission': False,
            'add_neb_continuum': True,
            'gas_logz': 0.0,
            'gas_logu': -2.0
        }
        myparams.update(nebular_params)
        # agn torus emission
        agn_params = {
            'add_agn': False,
            'fagn': 1e-4,
            # 'log_fagn': -4,
            'agn_tau': 5.0,
            # 'log_agn_tau': np.log10(5)
        }
        myparams.update(agn_params)
        return myparams

    def convert_myparams_model_params(self, params):
        """
        Convert our convenience parameter dictionary to Prospector required dictionary format

        Parameters
        ----------
        params : dict
            Our myparams format dictionary

        Returns
        -------
        model_params : dict
            Prospector model_params dictionary format
        """
        this_model_params = self.model_params.copy()
        this_model_params['imf_type'] = {'N': 1, 'isfree': False, 'init': self.imf_type}
        this_model_params['dust_type'] = {'N': 1, 'isfree': False, 'init': self.dust_type}
        # self.model_params['add_neb_emission'] = {'isfree': False, 'init': self.nebular}
        # self.model_params['add_agn'] = {'isfree': False, 'init': self.agn}
        this_model_params['sfh'] = {'N': 1, 'isfree': False, 'init': self.sfh_init}

        # fill in model_params with our own parameters, if possible
        for i, key in enumerate(list(params.keys())):
            try:
                this_model_params[key] = {'N': np.atleast_1d(params[key]).shape[0], 'isfree': False, 'init': params[key]}
            except:
                print(f"{key} can't be read")
                # pass

        tuniv = cosmology.age(params['zred']).value
        lumdist = cosmology.luminosity_distance(params['zred']).value
        this_model_params['lumdist'] = {"N": 1, "isfree": False, "init": lumdist, "units":"Mpc"}
        # convert dust_ratio to dust1
        this_model_params['dust1'] = {'N': 1, 'isfree': False, 'init': params['dust2']*params['dust_ratio']}
        # convert log parameters to regular parameters that Prospector recognizes
        # self.model_params['duste_gamma'] = {'N': 1, 'isfree': False, 'init': 10**params['duste_loggamma']}
        # self.model_params.pop('duste_loggamma')
        # self.model_params['fagn'] = {'N': 1, 'isfree': False, 'init': 10**params['log_fagn']}
        # self.model_params.pop('log_fagn')
        # self.model_params['agn_tau'] = {'N': 1, 'isfree': False, 'init': 10**params['log_agn_tau']}
        # self.model_params.pop('log_agn_tau')
        

        # SFH specific paremters translation
        if self.sfh_type == 'continuity_sfh':
            # use prospector function to create sfh bins related parameters
            this_model_params = adjust_continuity_agebins(this_model_params, tuniv=tuniv, nbins=params['nbins'])
            this_model_params.pop('nbins')
            # adjust_continuity_agebins will reset logsfr_ratio to all zeros, so read it in again
            this_model_params['logsfr_ratios']['init'] = np.array(params['logsfr_ratios'])
            this_model_params['logsfr_ratios']['isfree'] = False
            this_model_params['mass']['init'] = transforms.logsfr_ratios_to_masses(logmass=this_model_params['logmass']['init'], 
                                                                                   logsfr_ratios=this_model_params['logsfr_ratios']['init'], 
                                                                                   agebins=this_model_params['agebins']['init'])
        elif self.sfh_type == 'parametric_sfh':
            # convert tage_tuniv to tage
            tage = params['tage_tuniv'] * tuniv
            this_model_params['tage'] = {'N': 1, 'isfree': False, 'init': tage}
            # self.model_params.pop('tage_tuniv')
            # convert fage_burst to tburst
            tburst = tage * params['fage_burst']
            this_model_params['tburst'] = {'N': 1, 'isfree': False, 'init': tburst}
            # self.model_params.pop('fage_burst')
            # convert logmass to mass
            mass = 10**params['logmass']
            this_model_params['mass'] = {'N': 1, 'isfree': False, 'init': mass}
            # self.model_params.pop('logmass')
            # convert fage_trunc to sf_trunc
            sf_trunc = tage * params['fage_trunc']
            this_model_params['sf_trunc'] = {'N': 1, 'isfree': False, 'init': sf_trunc}
            this_model_params.pop('fage_trunc')
        
        return this_model_params


    def generate_spectra(self, myparams=None, model_params=None, wl_min=0.2, wl_max=6, filters=None, unit='uJy'):
        """
        Create spectra with prospector with input convenience parameter dictionary

        Parameters
        ----------
        params : dict
            Input parameters using our convenience format
        model_params : prospector dict
            Input parameters using prospector dictionary format
        wl_min : float, default=0.2
            Minimum wavelength output in micron
        wl_max : float, default=6.0
            Maximum wavelength output in micron
        filters : ndarray of shape (nfilt, 2, ns) or list of sedpy.observate.Filter
            Filters to evaluate observed spectra
            if using ndarray, filters[i][0] is wavelength in micron, filters[i][1] is response
            this will overwrite the previous provided filters when constructing the object
        unit : str, default='uJy'
            Output flux unit, can be 'uJy', 'mJy', 'Jy'

        Returns
        -------
        wavelength : ndarray of shape (nw, )
            rest-frame wavelength grid in micron
        spectra : ndarray of shape (nw, )
            model spectra, default in uJy
        flux_conv : ndarray of shape (nf, )
            convolved flux, default in uJy, will return 0.0 if filters are not provided
        mfrac : float
            surviving mass fraction
        """
        if unit == 'uJy':
            flux_factor = 1e6
        elif unit == 'mJy':
            flux_factor = 1e3
        elif unit == 'Jy':
            flux_factor = 1.0

        if myparams is not None:
            self.model_params = self.convert_myparams_model_params(myparams)
            zred = myparams['zred']
        elif model_params is not None:
            self.model_params = model_params
            zred = model_params['zred']['init']
        model = SpecModel(self.model_params)

        # in case user want to evaluate with a different filter, you can provided in this function to overwrite original filters
        if filters is not None:
            # TODO remove this part?
            obs = {}
            if type(filters) == np.ndarray:
                filterslist_sedpy = []
                filters_A = filters.copy()
                filters_A[:,0,:] *= 1e4
                filters_A[:,1,:][filters_A[:,1,:]<1e-5] = 1e-5
                Nfilt = filters_A.shape[0]
                filterslist_sedpy = []
                # np.trapz = np.trapezoid # sedpy annoyingly use np.trapz, which has been removed from newer numpy
                for i in range(Nfilt):
                    filterslist_sedpy.append(Filter(data=filters_A[i], kname=f'spherex_{str(i+1).zfill(3)}', min_trans=1e-6))                
                obs['filters'] = filterslist_sedpy
            elif type(filters) == list:
                obs['filters'] = filters

        else:
            filters = self.filters
            obs = self.obs

        # spectra, flux_conv, mfrac = model.predict({}, obs=obs, sps=self.sps)
        spectra, _, mfrac = model.predict({}, obs={}, sps=self.sps)    # TESTING
        
        # self.model = model
        wavelength = self.sps.wavelengths / 1e4
        if filters is not None:
            wavelength_ops = wavelength * (1+zred)
            flux_conv = f_convolve_filter(wavelength_ops, spectra, filters=filters)
        else:
            flux_conv = 0.0

        wavelength_mask = (wavelength >= wl_min) & (wavelength <= wl_max)
        wavelength = wavelength[wavelength_mask]
        spectra = spectra[wavelength_mask] * 3631 * flux_factor
        flux_conv = flux_conv * 3631 * flux_factor
        self.wavelength = wavelength
        self.spectra = spectra
        self.flux_conv = flux_conv
        self.mfrac = mfrac

        return wavelength, spectra, flux_conv, mfrac


