import gpflow
import numpy as np

# import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from .hapi import PYTIPS2017 #, molecularMass
from MATS.Utilities import molecularMass
from .tf_pcqsdhc import tf_pcqsdhc

# This recreates HTP_from_DF_select() for a single spectra
# It does NOT mix spectra, including multiple peaks, so what's in below, will remove one peak when fitting

# Class for a single, individual lineshape
# To model spectra, will need lots of these joined together with ComboMeanFunc


#Instead of loading constants from MATS.Utilities, define as dictionary
#More clear and avoids issues with variable scope and naming
constants = {'h' : 6.62607015e-27, #erg s https://physics.nist.gov/cgi-bin/cuu/Value?h|search_for=h as of 5/21/2020
             'c' : 29979245800, #cm/s # https://physics.nist.gov/cgi-bin/cuu/Value?c|search_for=c as of 5/21/2020
             'k' : 1.380649e-16, # erg / K https://physics.nist.gov/cgi-bin/cuu/Value?k as of 5/21/2020     
             'Na' : 6.02214076e23, # mol-1 https://physics.nist.gov/cgi-bin/cuu/Value?na as of 5/21/2020
             'cpa_atm' : (10*101325)**-1, #convert from cpa to atm  https://physics.nist.gov/cgi-bin/cuu/Value?stdatm|search_for=atmosphere as of 5/21/2020
            }
constants['c2'] =  (constants['h']*constants['c'])/constants['k']


class LineShape(gpflow.mean_functions.MeanFunction):
    def __init__(
        self,
        molec_id,
        iso,
        mole_fraction=1.0,
        n_data_sets=1,
        constraint_dict={},
        wing_method="wing_cutoff",
        cutoff=50,
        noise_scale_factor=1.0,
        fittable=True,
        **kwargs
    ):

        # Molecule/line of interest
        self.molec_id = molec_id
        self.iso = iso
        self.mole_frac = mole_fraction

        # Defaults, think diluent defaults are right for air?
        # Trainable parameters during optimization
        param_dict = {
            "nu": 1.0,
            "sw": 1.0,
            "gamma0": 0.02,
            "delta0": -0.01,
            "sd_gamma": 0.1,
            "sd_delta": 0.0,
            "nuvc": 0.0,
            "eta": 0.0,
            "y": 0.0,
        }
        # Non-trainable (cannot vary)
        non_param_dict = {
            "elower": 1.0,
            "sw_scale_fac": 1e-26,
            "n_gamma0": 0.63,
            "n_delta0": 5e-05,
            "n_gamma2": 0.63,
            "n_delta2": 0.0,
            "n_nuvc": 1.0,
        }
        # Update with passed keyword arguments
        for param_name in kwargs.keys():
            if param_name in param_dict.keys():
                param_dict[param_name] = kwargs[param_name]
            elif param_name in non_param_dict.keys():
                non_param_dict[param_name] = kwargs[param_name]
            else:
                raise ValueError("Argument %s not recognized." % param_name)

        # Keep track of parameters in dictionary rather than as object parameters
        self.params = {}

        # Set all parameters that are trainable
        for key, val in param_dict.items():
            if fittable:
                try:
                    low_bound, high_bound = tf.cast(constraint_dict[key], tf.float64)
                    self.params[key] = gpflow.Parameter(
                            np.array(val, dtype=np.float64),
                            dtype=tf.float64,
                            name=key,
                            trainable=True,
                            transform=tfp.bijectors.SoftClip(
                                low=low_bound, high=high_bound
                            ),
                        )

                except KeyError:
                    self.params[key] = gpflow.Parameter(np.array(val, dtype=np.float64),
                                                        dtype=tf.float64,
                                                        name=key,
                                                        trainable=True)
            else:
                self.params[key] = tf.constant(np.array(val, dtype=np.float64), dtype=tf.float64)

        # Set all other parameters
        for key, val in non_param_dict.items():
            self.params[key] = tf.constant(np.array(val, dtype=np.float64), dtype=tf.float64)

        # Defining how cutoffs handled for all lines
        self.wing_method = wing_method
        self.cutoff = cutoff

        # For stable training, best to make noise order 1, so accomplish with scaling factor
        self.noise_scaling = noise_scale_factor

        # Other stuff - seems to be constant and not change
        self.Tref = 296.0
        self.Pref = 1.0

    def get_dset_params(self, dInds, param_names):
        # Use indices of dataset for each data point to determine parameters for all data points
        # Returns tensors of same size as dInds for each adjustable parameter in param_names (list of strings)
        # Where the appropriate dataset-dependent parameter is used at each index
        param_list = []
        for param in param_names:
            this_param = self.params[param]
            if len(this_param.shape) > 0:
                if this_param.shape[0] > 1:
                    # Have different parameter for every dataset
                    param_list.append(tf.gather(this_param, dInds))
                else:
                    # Parameter that is 1D used for all datasets
                    param_list.append(tf.gather(this_param, np.zeros_like(dInds)))
            else:
                # Scalar parameter used for all datasets
                param_list.append(tf.fill(dInds.shape, this_param))
        return param_list

    def get_params_at_TP(self, T, P, nu, gamma0, delta0, sd_gamma, sd_delta, nuVC, eta):
        mass = molecularMass(self.molec_id, self.iso) #* 1.66053873e-27 * 1000
        gammaD = (
            np.sqrt(2 * constants['k'] * constants['Na'] * T * np.log(2) / mass)
            * nu / constants['c']
        )
        calc_gamma0 = gamma0 * (P / self.Pref) * ((self.Tref / T) ** self.params['n_gamma0'])
        shift0 = (delta0 + self.params['n_delta0'] * (T - self.Tref)) * (P / self.Pref)
        gamma2 = (
            sd_gamma * gamma0 * (P / self.Pref) * ((self.Tref / T) ** self.params['n_gamma2'])
        )
        shift2 = (sd_delta * delta0 + self.params['n_delta2'] * (T - self.Tref)) * (P / self.Pref)
        nuVC = nuVC * (P / self.Pref) * ((self.Tref / T) ** self.params['n_nuvc'])
        eta = eta
        return (gammaD, calc_gamma0, gamma2, shift0, shift2, nuVC, eta)

    def environmentdependency_intensity(self, T, nu, sw):
        sigmaT = np.array([PYTIPS2017(self.molec_id, self.iso, tval) for tval in T])
        sigmaTref = PYTIPS2017(self.molec_id, self.iso, self.Tref)
        # Taken from hapi.py and made compatible with tensorflow
        ch = tf.exp(-constants['c2'] * self.params['elower'] / T) * (1 - tf.exp(-constants['c2'] * nu / T))
        zn = tf.exp(-constants['c2'] * self.params['elower'] / self.Tref) * (
            1 - tf.exp(-constants['c2'] * nu / self.Tref)
        )
        LineIntensity = self.params['sw_scale_fac'] * sw * sigmaTref / sigmaT * ch / zn
        return LineIntensity

    def get_wave_cut_mask(self, wavenumbers, gammaD, gamma0, nu):
        if self.wing_method == "wing_cutoff":
            # Uses cutoff number of half-widths
            cut = (
                0.5346 * gamma0 + tf.math.sqrt(0.2166 * (gamma0 ** 2) + (gammaD ** 2))
            ) * self.cutoff
        else:
            # Applies to wing_wavenumbers method
            cut = self.cutoff
        mask = tf.math.logical_and(
            (wavenumbers >= (nu - cut)), (wavenumbers <= (nu + cut))
        )
        mask = tf.cast(mask, tf.float64)
        return mask

    def __call__(self, xTP):
        # First column is x, next is T, then P, and last is dataset indices
        xTP = np.array(xTP, dtype=np.float64)
        x = xTP[:, 0]
        T = xTP[:, 1]
        P = xTP[:, 2]
        dInds = np.array(xTP[:, 3], dtype=np.int32)

        mol_dens = (P / constants['cpa_atm']) / (constants['k'] * T)

        nu, sw, gamma0, delta0, sd_gamma, sd_delta, nuVC, eta, y = self.get_dset_params(
            dInds,
            [
                "nu",
                "sw",
                "gamma0",
                "delta0",
                "sd_gamma",
                "sd_delta",
                "nuvc",
                "eta",
                "y",
            ],
        )
        line_intensity = self.environmentdependency_intensity(T, nu, sw)
        y = y * (P / self.Pref)
        params = self.get_params_at_TP(
            T, P, nu, gamma0, delta0, sd_gamma, sd_delta, nuVC, eta
        )
        vals_real, vals_imag = tf_pcqsdhc(nu, *params, x)
        out = mol_dens * self.mole_frac * line_intensity * (vals_real + y * vals_imag)
        mask = self.get_wave_cut_mask(x, params[0], params[1], nu)
        out = mask * out * 1e06  # Make ppm/cm instead of just 1/cm
        out = out / self.noise_scaling
        out = tf.reshape(out, (-1, 1))
        return out


def lineshape_from_dataframe(frame, limit_factor_dict={}, line_kwargs={}):
    nu_list = frame.filter(regex=r"nu_*\d$").values.flatten().tolist()
    sw_list = frame.filter(regex=r"sw_*\d$").values.flatten().tolist()
    # Infer number of data sets from highest number associated with nu or sw
    n_dsets = len(nu_list)
    param_dict = {}
    vary_dict = {}
    vary_dict["nu"] = bool(
        np.sum(frame.filter(regex=r"nu_*\d_vary$").values)
    )  # Logical or, if any True, vary all
    vary_dict["sw"] = bool(
        np.sum(frame.filter(regex=r"sw_*\d_vary$").values, dtype=bool)
    )
    # Loop over parameters in dataframe excluding nu and sw
    for name, val in frame.iteritems():
        if (
            (name in ["molec_id", "local_iso_id"])
            or ("err" in name)
            or ("sw" in name and "scale_factor" not in name)
            or ("nu" in name and "VC" not in name)
        ):
            continue
        else:
            new_name = name.replace("_air", "").lower()
            if "y_296" in new_name:
                new_name = new_name.replace("y_296", "y")
            if "sw_scale_factor" in new_name:
                new_name = new_name.replace("sw_scale_factor", "sw_scale_fac")
            if "vary" in name:
                vary_dict[new_name.replace("_vary", "")] = bool(val)
            else:
                param_dict[new_name] = val
    #Separately loop over parameters to get constraints
    constraint_dict = {}
    for name, val in frame.iteritems():
        if ('vary' in name) or ("err" in name):
            continue
        else:
            new_name = name.replace("_air", "").lower()
            try:
                constraint_type, constraint_info = limit_factor_dict[name]
                if constraint_type == "magnitude":
                    constraint_dict[new_name] = (
                        val - constraint_info,
                        val + constraint_info,
                    )
                elif constraint_type == "factor":
                    constraint_dict[new_name] = (
                        val / constraint_info,
                        val * constraint_info,
                    )
            except KeyError:
                pass
    # Create our lineshape function
    lineshape = LineShape(
        frame["molec_id"],
        frame["local_iso_id"],
        n_data_sets=n_dsets,
        nu=nu_list,
        sw=sw_list,
        **param_dict,
        **line_kwargs,
        constraint_dict=constraint_dict
    )
    # Freeze somethings and let others vary
    for name, val in vary_dict.items():
        if isinstance(lineshape.params[name], gpflow.Parameter):
            gpflow.set_trainable(lineshape.params[name], val)
    return lineshape


class Etalon(gpflow.mean_functions.MeanFunction):
    def __init__(
        self, amplitude, period, phase, ref_wavenumber, noise_scale_factor=1.0, fittable=True
    ):
        # Note that if multidimensional, assumes different values for different datasets
        self.params = {}
        for name, val  in [['amp', amplitude],
                           ['period', period],
                           ['phase', phase],
                           ['ref_wave', ref_wavenumber]]:
            if fittable:
                self.params[name] = gpflow.Parameter(val, dtype=tf.float64, name=name, trainable=True)
            else:
                self.params[name] = val
        self.noise_scaling = noise_scale_factor

    def get_dset_params(self, dInds, param_names):
        # Use indices of dataset for each data point to determine parameters for all data points
        # Returns tensors of same size as dInds for each adjustable parameter in param_names (list of strings)
        # Where the appropriate dataset-dependent parameter is used at each index
        param_list = []
        for param in param_names:
            this_param = self.params[param]
            if len(this_param.shape) > 0:
                if this_param.shape[0] > 1:
                    # Have different parameter for every dataset
                    param_list.append(tf.gather(this_param, dInds))
                else:
                    # Parameter that is 1D used for all datasets
                    param_list.append(tf.gather(this_param, np.zeros_like(dInds)))
            else:
                # Scalar parameter used for all datasets
                param_list.append(tf.fill(dInds.shape, this_param))
        return param_list

    def __call__(self, xTP):
        xTP = np.array(xTP, dtype=np.float64)
        wavenumbers = xTP[:, 0]
        dInds = np.array(xTP[:, 3], dtype=np.int32)
        amps, periods, phases, ref_waves = self.get_dset_params(dInds, list(self.params.keys()))
        etalon_model = amps * tf.math.sin(
            (2 * np.pi * periods) * (wavenumbers - ref_waves) + phases
        )
        etalon_model = etalon_model / self.noise_scaling
        etalon_model = tf.reshape(etalon_model, (-1, 1))
        return etalon_model


#Should add function to create etalons from baseline parameter list
#Though with various kernels, should be able to take care of etalons without specifying...


class ComboMeanFunc(gpflow.mean_functions.MeanFunction):
    def __init__(self, func_list):
        self.mean_funcs = func_list

    def __call__(self, x_input):
        out = tf.reduce_sum([f(x_input) for f in self.mean_funcs], axis=0)
        return out
