import gpflow
import numpy as np

# import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from MATS.codata import CONSTANTS
from MATS.hapi import PYTIPS2017  # , molecularMass
from MATS.utilities import molecularMass

from .tf_pcqsdhc import tf_pcqsdhc

# Instead of loading constants from MATS.Utilities, define as dictionary
# More clear and avoids issues with variable scope and naming
# constants = {'h' : 6.62607015e-27, #erg s https://physics.nist.gov/cgi-bin/cuu/Value?h|search_for=h as of 5/21/2020
#              'c' : 29979245800, #cm/s # https://physics.nist.gov/cgi-bin/cuu/Value?c|search_for=c as of 5/21/2020
#              'k' : 1.380649e-16, # erg / K https://physics.nist.gov/cgi-bin/cuu/Value?k as of 5/21/2020
#              'Na' : 6.02214076e23, # mol-1 https://physics.nist.gov/cgi-bin/cuu/Value?na as of 5/21/2020
#              'cpa_atm' : (10*101325)**-1, #convert from cpa to atm  https://physics.nist.gov/cgi-bin/cuu/Value?stdatm|search_for=atmosphere as of 5/21/2020
#             }
# constants['c2'] =  (constants['h']*constants['c'])/constants['k']


class SpectralDataInfo(gpflow.base.Module):
    def __init__(self, constraint_dict={}, fittable=True, **kwargs):

        # Trainable parameters
        param_dict = {
            "mole_frac": 1.0,
            "x_shift": 0.0,
        }

        # Non-trainable
        non_param_dict = {
            "nominal_temp": 273,
            "abun_ratio": 1.0,
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
                        trainable=False,
                        transform=tfp.bijectors.SoftClip(
                            low=low_bound, high=high_bound
                        ),
                    )

                except KeyError:
                    self.params[key] = gpflow.Parameter(
                        np.array(val, dtype=np.float64),
                        dtype=tf.float64,
                        name=key,
                        trainable=False,
                    )
            else:
                self.params[key] = tf.constant(
                    np.array(val, dtype=np.float64), dtype=np.float64
                )

        # Set all other parameters
        for key, val in non_param_dict.items():
            self.params[key] = tf.constant(
                np.array(val, dtype=np.float64), dtype=tf.float64
            )

    def mole_frac(self):
        return self.params["mole_frac"]

    def nominal_temp(self):
        return self.params["nominal_temp"]

    def x_shift(self):
        return self.params["x_shift"]

    def abun_ratio(self):
        return self.params["abun_ratio"]


class LineMixing(gpflow.base.Module):
    def __init__(
        self,
        nom_temps,
        y_vals,
        constraint_dict={},
        fittable=True,
    ):
        if len(nom_temps) != len(y_vals):
            raise ValueError(
                "Must have same number of nominal temperatures and y values."
            )
        self.y = {}
        for t, y in zip(nom_temps, y_vals):
            if fittable:
                try:
                    low_bound, high_bound = tf.cast(
                        constraint_dict["y_" + str(t)], tf.float64
                    )
                    self.y[t] = gpflow.Parameter(
                        y,
                        dtype=tf.float64,
                        name="y_" + str(t),
                        trainable=False,
                        transform=tfp.bijectors.SoftClip(
                            low=low_bound, high=high_bound
                        ),
                    )
                except KeyError:
                    self.y[t] = gpflow.Parameter(
                        y, dtype=tf.float64, name="y_" + str(t), trainable=False
                    )
            else:
                self.y[t] = tf.constant(y, dtype=tf.float64)

    def __call__(self, nom_temp):
        if np.ndim(nom_temp) > 0:
            out = [self.y[t] for t in nom_temp]
        else:
            out = [self.y[nom_temp]]
        out = tf.stack(out)
        return out


def linemix_from_dataframe(frame, limit_factor_dict={}, linemix_kwargs={}):
    nom_temps = []
    y_vals = []
    vary_dict = {}
    constraint_dict = {}
    # Loop over frames, selecting out those with y info
    for name, val in frame.iteritems():
        if ("y_" in name) and ("err" not in name):
            new_name = name.replace("_air", "").lower()
            if "vary" in name:
                vary_dict[float(new_name.replace("_vary", "").split("_")[-1])] = bool(
                    val
                )
            else:
                nom_temps.append(float(new_name.split("_")[-1]))
                y_vals.append(val)
                # Try to get constraints
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
                    # Otherwise assume bounds explicitly provided
                    else:
                        constraint_dict[new_name] = constraint_info
                except KeyError:
                    pass
        else:
            continue

    # Create our linemixing object
    linemix = LineMixing(
        nom_temps, y_vals, **linemix_kwargs, constraint_dict=constraint_dict
    )
    # Freeze some things and let others vary
    for name, val in vary_dict.items():
        if isinstance(linemix.y[name], gpflow.Parameter):
            gpflow.set_trainable(linemix.y[name], val)
    return linemix


class Base_Mean_Func(gpflow.mean_functions.MeanFunction):
    def __init__(self):
        super().__init__()

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

    def __call__(self):
        raise NotImplementedError(
            "Need to define __call__ method if work off this base class."
        )


# This recreates HTP_from_DF_select() for a single spectra
# Class for a single, individual lineshape
# To model spectra, will need lots of these joined together with ComboMeanFunc
class LineShape(Base_Mean_Func):
    def __init__(
        self,
        molec_id,
        iso,
        dset_list=[SpectralDataInfo()],
        linemix=None,
        constraint_dict={},
        wing_method="wing_cutoff",
        cutoff=50,
        noise_scale_factor=1.0,
        fittable=True,
        **kwargs
    ):

        super().__init__()

        # Molecule/line of interest
        self.molec_id = molec_id
        self.iso = iso

        # Specify list of dataset information
        self.dset_list = dset_list

        # Take care of line mixing information if not specified
        if linemix is None:
            self.linemix = LineMixing(
                [dset.nominal_temp() for dset in dset_list], [0.0] * len(dset_list)
            )
        else:
            self.linemix = linemix

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
            "n_gamma0": 0.63,
            "n_delta0": 5e-05,
            "n_gamma2": 0.63,
            "n_delta2": 0.0,
            "n_nuvc": 1.0,
        }
        # Non-trainable (cannot vary)
        non_param_dict = {
            "elower": 1.0,
            "sw_scale_fac": 1e-26,
        }
        # Update with passed keyword arguments
        for param_name in kwargs.keys():
            if param_name in param_dict.keys():
                param_dict[param_name] = kwargs[param_name]
            elif param_name in non_param_dict.keys():
                non_param_dict[param_name] = kwargs[param_name]
            else:
                print("WARNING: Argument %s not recognized, so ignoring." % param_name)
                continue

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
                        trainable=False,
                        transform=tfp.bijectors.SoftClip(
                            low=low_bound, high=high_bound
                        ),
                    )

                except KeyError:
                    self.params[key] = gpflow.Parameter(
                        np.array(val, dtype=np.float64),
                        dtype=tf.float64,
                        name=key,
                        trainable=False,
                    )
            else:
                self.params[key] = tf.constant(
                    np.array(val, dtype=np.float64), dtype=tf.float64
                )

        # Set all other parameters
        for key, val in non_param_dict.items():
            self.params[key] = np.array(
                val, dtype=np.float64
            )  # Avoid tf - never optimized

        # Defining how cutoffs handled for all lines
        self.wing_method = wing_method
        self.cutoff = cutoff

        # For stable training, best to make noise order 1, so accomplish with scaling factor
        self.noise_scaling = noise_scale_factor

        # Other stuff - seems to be constant and not change
        self.Tref = 296.0
        self.Pref = 1.0

    def get_params_at_TP(self, T, P, nu, gamma0, delta0, sd_gamma, sd_delta, nuVC, eta):
        mass = molecularMass(self.molec_id, self.iso)  # * 1.66053873e-27 * 1000
        gammaD = (
            np.sqrt(2 * CONSTANTS["k"] * CONSTANTS["Na"] * T * np.log(2) / mass)
            * nu
            / CONSTANTS["c"]
        )
        calc_gamma0 = (
            gamma0 * (P / self.Pref) * ((self.Tref / T) ** self.params["n_gamma0"])
        )
        shift0 = (delta0 + self.params["n_delta0"] * (T - self.Tref)) * (P / self.Pref)
        gamma2 = (
            sd_gamma
            * gamma0
            * (P / self.Pref)
            * ((self.Tref / T) ** self.params["n_gamma2"])
        )
        shift2 = (sd_delta * delta0 + self.params["n_delta2"] * (T - self.Tref)) * (
            P / self.Pref
        )
        nuVC = nuVC * (P / self.Pref) * ((self.Tref / T) ** self.params["n_nuvc"])
        eta = eta
        return (gammaD, calc_gamma0, gamma2, shift0, shift2, nuVC, eta)

    def environmentdependency_intensity(self, T, nu, sw):
        sigmaT = np.array([PYTIPS2017(self.molec_id, self.iso, tval) for tval in T])
        sigmaTref = PYTIPS2017(self.molec_id, self.iso, self.Tref)
        # Taken from hapi.py and made compatible with tensorflow
        ch = tf.exp(-CONSTANTS["c2"] * self.params["elower"] / T) * (
            1 - tf.exp(-CONSTANTS["c2"] * nu / T)
        )
        zn = tf.exp(-CONSTANTS["c2"] * self.params["elower"] / self.Tref) * (
            1 - tf.exp(-CONSTANTS["c2"] * nu / self.Tref)
        )
        LineIntensity = self.params["sw_scale_fac"] * sw * sigmaTref / sigmaT * ch / zn
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

        # Each shift is constant across all spectra working with the same datasets
        # So have SpectralDataInfo objects outside that are provided as a list to LineShape
        # The dInds reference the indices of the SpectralDataInfo objects in the list
        x_shift = tf.gather(
            [tf.convert_to_tensor(dset.x_shift()) for dset in self.dset_list], dInds
        )
        x = x + x_shift

        mol_dens = (P / CONSTANTS["cpa_atm"]) / (CONSTANTS["k"] * T)

        nu, sw, gamma0, delta0, sd_gamma, sd_delta, nuVC, eta = self.get_dset_params(
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
            ],
        )
        line_intensity = self.environmentdependency_intensity(T, nu, sw)

        # Line mixing terms are defined not by dataset or lineshape, but by nominal temperature
        # So have independent object outside each
        # Workflow is to use dInds to get nominal temperatures for each dataset in dset_list
        # Then call linemix, which is a potentially shared LineMixing object to get y for each
        nom_temps = np.take([dset.nominal_temp() for dset in self.dset_list], dInds)
        y = self.linemix(nom_temps) * (P / self.Pref)

        params = self.get_params_at_TP(
            T, P, nu, gamma0, delta0, sd_gamma, sd_delta, nuVC, eta
        )
        vals_real, vals_imag = tf_pcqsdhc(nu, *params, x)

        mole_frac = tf.gather(
            [tf.convert_to_tensor(dset.mole_frac()) for dset in self.dset_list], dInds
        )
        abun_ratio = tf.gather(
            [tf.convert_to_tensor(dset.abun_ratio()) for dset in self.dset_list], dInds
        )
        out = (
            mol_dens
            * mole_frac
            * abun_ratio
            * line_intensity
            * (vals_real + y * vals_imag)
        )

        mask = self.get_wave_cut_mask(x, params[0], params[1], nu)
        out = mask * out * 1e06  # Make ppm/cm instead of just 1/cm

        out = out / self.noise_scaling
        out = tf.reshape(out, (-1, 1))
        return out


def lineshape_from_dataframe(frame, limit_factor_dict={}, line_kwargs={}):
    nu_list = frame.filter(regex=r"nu_\d*$").values.flatten().tolist()
    sw_list = frame.filter(regex=r"sw_\d*$").values.flatten().tolist()
    # Make sure nu_list and sw_list are not empty, otherwise will overide default with empty
    if len(nu_list) == 0:
        nu_list = frame.filter(regex=r"nu$").values.flatten().tolist()
    if len(sw_list) == 0:
        sw_list = frame.filter(regex=r"sw$").values.flatten().tolist()

    # Infer number of data sets from highest number associated with nu or sw
    # variable never used, so comment out
    # n_dsets = np.max([len(nu_list), len(sw_list)])

    param_dict = {}
    vary_dict = {}
    vary_dict["nu"] = bool(
        np.sum(frame.filter(regex=r"nu_\d*_vary$").values)
        + np.sum(frame.filter(regex=r"nu_vary$").values)
    )  # Logical or, if any True, vary all
    vary_dict["sw"] = bool(
        np.sum(frame.filter(regex=r"sw_\d*_vary$").values, dtype=bool)
        + np.sum(frame.filter(regex=r"sw_vary$").values)
    )

    # Loop over parameters in dataframe excluding nu and sw
    for name, val in frame.iteritems():
        if (
            (name in ["molec_id", "local_iso_id"])
            or ("err" in name)
            or ("sw" in name and "scale_factor" not in name)
            or ("nu" in name and "VC" not in name)
            or ("y_" in name)
        ):
            continue
        else:
            new_name = name.replace("_air", "").lower()
            if "sw_scale_factor" in new_name:
                new_name = new_name.replace("sw_scale_factor", "sw_scale_fac")
            if "vary" in name:
                vary_dict[new_name.replace("_vary", "")] = bool(val)
            else:
                param_dict[new_name] = val
    # Separately loop over parameters to get constraints
    constraint_dict = {}
    for name, val in frame.iteritems():
        if ("vary" in name) or ("err" in name):
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


class Etalon(Base_Mean_Func):
    def __init__(
        self,
        amplitude,
        period,
        phase,
        ref_wavenumber,
        noise_scale_factor=1.0,
        fittable=True,
    ):
        super().__init__()
        # Note that if multidimensional, assumes different values for different datasets
        self.params = {}
        for name, val in [
            ["amp", amplitude],
            ["period", period],
            ["phase", phase],
            ["ref_wave", ref_wavenumber],
        ]:
            if fittable:
                self.params[name] = gpflow.Parameter(
                    val, dtype=tf.float64, name=name, trainable=False
                )
            else:
                self.params[name] = val
        self.noise_scaling = noise_scale_factor

    def __call__(self, xTP):
        xTP = np.array(xTP, dtype=np.float64)
        wavenumbers = xTP[:, 0]
        dInds = np.array(xTP[:, 3], dtype=np.int32)
        amps, periods, phases, ref_waves = self.get_dset_params(
            dInds, list(self.params.keys())
        )
        etalon_model = amps * tf.math.sin(
            (2 * np.pi * periods) * (wavenumbers - ref_waves) + phases
        )
        etalon_model = etalon_model / self.noise_scaling
        etalon_model = tf.reshape(etalon_model, (-1, 1))
        return etalon_model


# Should add function to create etalons from baseline parameter list
# Though with various kernels, should be able to take care of etalons without specifying...
# (and should also create baseline objects, too...)


class Baseline(Base_Mean_Func):
    def __init__(
        self, c0, c1, c2, ref_wavenumber, noise_scale_factor=1.0, fittable=True
    ):
        super().__init__()
        # Note that if multidimensional, assumes different values for different datasets
        self.params = {}
        for name, val in [
            ["c0", c0],
            ["c1", c1],
            ["c2", c2],
            ["ref_wave", ref_wavenumber],
        ]:
            if fittable:
                self.params[name] = gpflow.Parameter(
                    val, dtype=tf.float64, name=name, trainable=False
                )
            else:
                self.params[name] = val
        self.noise_scaling = noise_scale_factor

    def __call__(self, xTP):
        xTP = np.array(xTP, dtype=np.float64)
        wavenumbers = xTP[:, 0]
        dInds = np.array(xTP[:, 3], dtype=np.int32)
        c0, c1, c2, ref_waves = self.get_dset_params(dInds, list(self.params.keys()))
        baseline_model = tf.math.polyval([c2, c1, c0], wavenumbers - ref_waves)
        baseline_model = baseline_model / self.noise_scaling
        baseline_model = tf.reshape(baseline_model, (-1, 1))
        return baseline_model


class ComboMeanFunc(gpflow.mean_functions.MeanFunction):
    def __init__(self, func_list):
        self.mean_funcs = func_list

    def __call__(self, x_input):
        out = tf.reduce_sum([f(x_input) for f in self.mean_funcs], axis=0)
        return out
