"""
Created on Wed Jun 9

@author: Alexander Spokoinyi
"""
import logging
import numpy as np
from scipy.stats import variation, sem
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import erfcx
from tqdm import tqdm

from predCoding import *


# define some standard input functions
def input_cos(t, freq=0.1, ampl=0.1): return ampl * np.cos(freq * t)


def input_0(t): return 0 * t


time_cst = 20


def input_cst(t, strength=1): return strength / time_cst + 0 * t


######## Evaluate simulation results ########
def mean_est_sim(pc_sim, t_stable=None):
    """
    Compute mean readout from empirical simulation
    Args:
        pc_sim: pc_model
        t_stable: duration of initial transient which should be discarded (to average over stationary activity)

    Returns: float

    """
    if t_stable == None:
        t_stable = int(7.5 * 1 / (pc_sim.lambd * pc_sim.t_step))
    return np.mean(pc_sim.predSignal[0, t_stable:])


def var_readout_sim(pc_sim, t_stable=None):
    """

    Args:
        pc_sim: pc_model
        t_stable: duration of initial transient which should be discarded

    Returns: float

    """
    if t_stable == None:
        t_stable = int(7.5 * 1 / (pc_sim.lambd * pc_sim.t_step))
    return np.var(pc_sim.predSignal[0, t_stable:])


def count_synchron_spikes(sim):
    """

    Args:
        sim: pc_model - model with completed simulation

    Returns: int - number of synchronous spikes

    """
    spikes_over_time = np.sum(sim.spikes, 0)
    return np.sum(spikes_over_time > 1)


def mean_fir_rates(sim, cutoff=None):
    """
    Compute mean firing rates over stationary activity
    Args:
        sim: pc_model
        cutoff: float - duration of initial transient which should be discarded

    Returns: float

    """
    if cutoff == None:
        cutoff = int(5 * 1 / (sim.lambd))
    cutoff_i = int(cutoff / sim.t_step)
    neurons_plus = np.where(sim.weights[0, :] > 0)[0]
    neurons_minus = np.where(sim.weights[0, :] < 0)[0]
    norm_fact_plus = neurons_plus.size * (sim.T - cutoff)
    norm_fact_minus = neurons_minus.size * (sim.T - cutoff)
    fir_rate_plus = np.sum(sim.spikes[neurons_plus, cutoff_i:]) / norm_fact_plus
    fir_rate_minus = np.sum(sim.spikes[neurons_minus, cutoff_i:]) / norm_fact_minus
    # return values in Hz
    return [fir_rate_plus * 1000, fir_rate_minus * 1000]


def cv(pc_sim, cutoff=100, pos_only=True):
    """
    Compute coefficient of variation over stationary network activity
    Args:
        pc_sim: pc_model
        cutoff: float - duration of initial transient which should be discarded
        pos_only: bool - whether to compute cv only over the neurons with positive weight

    Returns: float

    """
    res_cv = np.zeros(pc_sim.nNeur)
    if pos_only:
        range_n = int(pc_sim.nNeur / 2 + 1)
    else:
        range_n = pc_sim.nNeur + 1
    for n in range(1, range_n):
        if np.sum(pc_sim.spike_neur == n) > 1:
            spike_times_n = pc_sim.spike_times[pc_sim.spike_neur == n] * pc_sim.t_step
            spike_times_n = spike_times_n[spike_times_n > cutoff]
            isi_n = spike_times_n[1:] - spike_times_n[:-1]
            if isi_n.size: res_cv[n - 1] = variation(isi_n)
    res_cv = res_cv[res_cv > 0]
    if not res_cv.size:
        logging.warning('In: cv(). Message: No neurons with more than 1 spike after cutoff. Returning \'None\'.')
        return None
    return np.mean(res_cv)


########### Mean-Field Predictions ############

def first_mean_rel(pc_sim, mean_q_grid):
    """
    First analytical relation between the means of mesoscopic variable q and the mean readout
    Args:
        pc_sim: pc_model
        mean_q_grid: np.ndarray - grid for the potential means of q over which to evaluate the first relation

    Returns: np.ndarray - grid with mean readout values corresponding to the q values given as input

    """
    time_cst = 1 / pc_sim.lambd
    mean_x_vals = time_cst * (pc_sim.input_s[0, 0] - 1 / pc_sim.connStrength * mean_q_grid)
    return mean_x_vals


def second_mean_rel(pc_sim, mean_q_grid):
    """
        Second analytical relation between the means of mesoscopic variable q and the mean readout
    Args:
        pc_sim: pc_model
        mean_q_grid: grid for the potential means of q over which to evaluate the second relation

    Returns: np.ndarray - grid with mean readout values corresponding to the q values given as input

    """
    time_cst = 1 / pc_sim.lambd
    # v1=lif_transferFct(mean_q_grid,pc_sim.lambd,pc_sim.sigma_V,pc_sim.Vthr,pc_sim.Vreset)
    # v2=lif_transferFct(-mean_q_grid,pc_sim.lambd,pc_sim.sigma_V,pc_sim.Vthr,pc_sim.Vreset)
    v1 = pc_sim.lif_transferFct(mean_q_grid)
    v2 = pc_sim.lif_transferFct(-mean_q_grid)
    return time_cst * pc_sim.connStrength / 2 * (v1 - v2)


def obj_fct_mean_q(q, pc_sim, s=None):
    """
    Helper function to numerically compute analytical mean of mesoscopic variable q
    Args:
        q: float - input q
        pc_sim: pc_model
        s: float - input strength (can be different from the one used in pc_sim

    Returns: float - result from quadratic objective function used for optimization

    """
    if s == None: s = pc_sim.input_s[0, 0]
    expectation_term = 0.5 * (pc_sim.lif_transferFct(q) - pc_sim.lif_transferFct(-q))
    mean_est = pc_sim.connStrength * (s - pc_sim.connStrength * expectation_term)
    return (q - mean_est) ** 2


def calc_mf_means(pc_sim, s=None):
    """
    Compute analytical results for the mean readout corresponding to a stationary signal
    Args:
        pc_sim: pc_model
        s: float - input strength

    Returns: float

    """
    if s == None: s = pc_sim.input_s[0, 0]
    time_cst = 1 / pc_sim.lambd
    mean_q = minimize(obj_fct_mean_q, 0, args=(pc_sim, s)).x
    mean_readout = time_cst * (s - 1 / pc_sim.connStrength * mean_q)
    return [mean_readout, mean_q]


def _lif_transferFct(mu, pc_sim):
    return pc_sim.lif_transferFct(mu / pc_sim.lambd)


def lif_tr_fct_der(mu, lambd, sigma_V, Vthr, Vreset, Vrest=0):
    """
    Derivative of LIF transfer function

    """
    noiseScale = np.sqrt(2) * sigma_V
    int_limit1, int_limit2 = [(1 / lambd * mu - Vthr) / noiseScale,
                              (1 / lambd * mu - Vreset) / noiseScale]
    if erfcx(int_limit1) and erfcx(int_limit2):
        int_res, _ = quad(func=erfcx, a=int_limit1, b=int_limit2)
    elif mu > 0:
        return lif_tr_fct_der(10, lambd, sigma_V, Vthr, Vreset, Vrest=0)
    elif mu < 0:
        return 0
    res = -1 / (np.sqrt(np.pi) * noiseScale) * (erfcx(int_limit2) - erfcx(int_limit1)) * \
          int_res ** (-2)

    return res


v_lif_tr_fct_der = np.vectorize(lif_tr_fct_der)


def _lif_tr_fct_der(mu, pc_sim):
    noiseScale = np.sqrt(2) * pc_sim.sigma_V
    int_limit1, int_limit2 = [(1 / pc_sim.lambd * mu - pc_sim.Vthr) / noiseScale,
                              (1 / pc_sim.lambd * mu - pc_sim.Vreset) / noiseScale]
    if erfcx(int_limit1) and erfcx(int_limit2):
        int_res, _ = quad(func=erfcx, a=int_limit1, b=int_limit2)
    elif mu > 0:
        return _lif_tr_fct_der(10, pc_sim)
    elif mu < 0:
        return 0
    res = -1 / (np.sqrt(np.pi) * noiseScale) * (erfcx(int_limit2) - erfcx(int_limit1)) * \
          int_res ** (-2)

    return res


_v_lif_tr_fct_der = np.vectorize(_lif_tr_fct_der)


def var_readout_anal(pc_mod, tau_h=None, s=None):
    """
    Compute analytical result for the readout variance (given a stationary input)
    Args:
        pc_mod: pc_model
        tau_h: time-scale
        s: input strength

    Returns: float

    """
    if s == None: s = pc_mod.input_s[0, 0]
    if tau_h == None: tau_h = pc_mod.tau
    N = pc_mod.nNeur
    mean_q = calc_mf_means(pc_mod, s)[1][0]
    nonlin_gain = 1 / 2 * (_lif_tr_fct_der(mean_q, pc_mod) + _lif_tr_fct_der(-mean_q, pc_mod))
    nonlin_drive = 1 / 2 * (pc_mod.lif_transferFct(mean_q) + pc_mod.lif_transferFct(-mean_q))
    effect_gain = 1 + pc_mod.connStrength ** 2 * nonlin_gain
    var_numerator = pc_mod.connStrength ** 2 * (1 + tau_h * pc_mod.lambd * effect_gain) * nonlin_drive
    var_denominator = 2 * N * effect_gain * (effect_gain + tau_h * pc_mod.lambd) * pc_mod.lambd
    return var_numerator / var_denominator


###### Estimating features over multiple simulations #########

def format_res_from_nSims(res_nSims, msg_spec='', return_sem=False, **simArgs):
    """
    Given an array where each entry represents a computational feature (such as mean readout)
    obtained from a single simulation, computes the average over all simulations
    and checks whether the standard error is below 5%. If it is not, a warning message is displayed.

    Args:
        res_nSims: np.ndarray - array with values to average over
        msg_spec: warning message to display if standard error is high
        return_sem: bool - if True, returns the standard error of the mean
        simArgs: simulation parameters for specification in warning message (if msg_spec is not given)

    Returns: float OR Tuple[float, float] - depending on whether return_sem is True or not

    """
    res_mean, res_sem = np.mean(res_nSims), sem(res_nSims)
    if res_sem > np.abs(res_mean) / 25:
        if msg_spec:
            specs_str = msg_spec
        elif simArgs:
            specs_str = f' for {simArgs}'
        else:
            specs_str = ''
        logging.warning('High standard error: SEM is {:.0%} of the mean'.format(res_sem / np.abs(res_mean)) + specs_str)
    if return_sem:
        return res_mean, res_sem
    else:
        return res_mean

def avs_over_sims_lif(target_funcs, nSims=10, target_dims=1, warning_spec='',
                      max_nSynchr=20, **simArgs):
    """
    Helper function: Average the result of any (scalar) function (which takes a LIF model simulation as argument) over
    multiple simulations
    Args:
        target_func: function - function to be computed
        nSims: int - number of simulations to average over
        target_dim: int - output dimensionality of target_func
        warning_spec: string - message to be displayed in warning when error is above 5%

    Returns: float or Tuple[float] - average value or each output dimension of the target function
    """

    if type(target_funcs) is not list:
        target_funcs = [target_funcs]
        target_dims = [target_dims]
    res_list = []
    nFuncs = len(target_funcs)
    for i in range(nFuncs):
        res_list.append(np.zeros([nSims, target_dims[i]]))
    n = 0
    nSkipped = 0
    while n < nSims:
        sim_n = spikeMod_sim(sparse=True, **simArgs)
        if np.sum(np.sum(sim_n.spikes[:, int(100 / sim_n.t_step):], 0) > max_nSynchr) > 0:
            if nSkipped > 10:
                logging.warning(
                    f'Skipped over 10 sims due to synchronisation. Continuing in asynchr regime ' + f'({warning_spec})')
                simArgs['asynchr'] = True
                nSkipped = 0
            else:
                nSkipped = nSkipped + 1
                continue

        for j, target_func in enumerate(target_funcs):
            res_list[j][n, :] = target_func(sim_n)
        n = n + 1

    for i, func_i in enumerate(target_funcs):
        if nFuncs > 1:
            spec = warning_spec + f' ({str(func_i).split()[1]})'
        else:
            spec = warning_spec
        res_list[i] = np.squeeze(np.apply_along_axis(format_res_from_nSims, 0, res_list[i],
                                                     msg_spec=spec, **simArgs))
    if nSkipped > 0: logging.warning(f'Skipped {nSkipped} sims due to synchronisation' + warning_spec)
    return res_list

def avs_over_sims_p(target_funcs, nSims=10, target_dims=1,
                    base_model=None, warning_spec='', max_nSynchr=20, **simArgs):
    """
    Helper function: Mean estimates of several functions (which takes a Poisson model simulation as argument) over multiple simulations
    Args:

        target_funcs: List[function] - functions to be computed
        nSims: int - number of simulations to average over
        target_dims: List[int] - output dimensionality of target_func
        base_model: pc_model - initialised Poisson model with pre-computed lookup table for the transfer function
        warning_spec: string - message to be displayed in warning when error is above 5%

    Returns: List[float or Tuple[float]] - List with average results for all target functions
    """
    if base_model == None:
        base_model = rateMod_sim_fast(manualSim=True, sparse=True, **simArgs)
        dynamic_args = {}
    elif any(elem not in ['input_fct', 'T', 't_step', 'tau', 'asynchr'] for elem in simArgs):
        raise NotImplementedError  # xxx
    else:
        dynamic_args = simArgs
    res_list = []
    if type(target_funcs) is not list:
        target_funcs = [target_funcs]
        target_dims = [target_dims]
    nFuncs = len(target_funcs)
    for i in range(nFuncs):
        res_list.append(np.zeros([nSims, target_dims[i]]))
    n = 0
    nSkipped = 0
    while n < nSims:
        base_model.sim(sparse=True, **dynamic_args)
        sim_n = base_model  # xxx
        if np.sum(np.sum(sim_n.spikes[:, int(100 / sim_n.t_step):], 0) > max_nSynchr) > 0:
            nSkipped = nSkipped + 1
            continue
        for i, target_func in enumerate(target_funcs):
            res_list[i][n, :] = target_func(base_model)
        n = n + 1
    for i, func_i in enumerate(target_funcs):
        if nFuncs > 1:
            spec = warning_spec + f' ({str(func_i).split()[1]})'
        else:
            spec = warning_spec
        res_list[i] = np.squeeze(np.apply_along_axis(format_res_from_nSims, 0, res_list[i],
                                                     msg_spec=spec, **simArgs))

    if nSkipped > 0: logging.warning(f'Skipped {nSkipped} sims due to synchronisation' + warning_spec)
    return res_list


def av_fcts_over_1Dgrid_lif(funcs, par, vals, nSims=10, target_dims=1, **simArgs):
    """
    Mean estimation of several LIF model features varying one parameter over a given value grid
    Args:
        funcs: List[function] - functions to be computed
        par: List[float] - simulation parameter that is evaluated
        vals: List[float] - values for par ver which to evaluate the target functions
        nSims: int - number of simulations to average over
        target_dims: int - output dimensionalities of the target functions
        simArgs: Additional simulation arguments (see spikeMod_sim for more details)

    Returns: np.ndarray - two-dimensional array with dimensionality determined by target_dim
                            and the length of vals
    """
    if type(funcs) is not list:
        funcs = [funcs]
        target_dims = [target_dims]
    res = [np.zeros([len(vals), target_dims[i]]) for i in range(len(funcs))]

    for i, par_i in tqdm(enumerate(vals)):
        if par == 'x':
            simArgs['input_fct'] = lambda t: input_cst(t, par_i)
        else:
            exec(f"simArgs['{par}']={par_i}")
        res_raw = avs_over_sims_lif(funcs, nSims,
                                    target_dims=target_dims, warning_spec=f' for {par}={par_i}',
                                    **simArgs)
        for k, res_k in enumerate(res_raw):
            res[k][i, :] = res_k

    if len(res) == 1:
        return res[0]
    else:
        return res


def av_fct_over_1Dgrid_p(func, par, vals, nSims=10, target_dim=1, base_model=None, **simArgs):
    """
    Mean estimation of several Poisson model features varying one parameter over a given value grid

    Returns: and the length of vals
    Args:
        func: List[function] - function to be computed
        par: List[float] - simulation parameter that is evaluated
        vals: List[float] - values for par ver which to evaluate the target function
        nSims: int - number of simulations to average over
        target_dim: int - dimensionality of target function func
        base_model: rateMod_sim_fast - optional: Poisson simulation instance with pre-computed lookup table for LIF transfer function
        **simArgs: other (fixed) simulation arguments

    Returns:  np.ndarray - array with the same number of elements as in vals
    """
    res = np.zeros([len(vals), target_dim])
    for i, par_i in tqdm(enumerate(vals)):
        if par == 'x':
            simArgs['input_fct'] = lambda t: input_cst(t, par_i)
        else:
            exec(f"simArgs['{par}']={par_i}")
        spec_str = f' for {par}: {par_i}'
        res[i, :] = avs_over_sims_p(func, nSims,
                                    target_dims=target_dim, base_model=base_model, warning_spec=spec_str,
                                    **simArgs)[0]
    return np.squeeze(res)

def av_fct_over_2Dgrid_p(funcs, par1, vals1, par2, vals2, nSims=10,
                         target_dims=1, base_model=None, **simArgs):
    """"
    Mean estimates of several functions (which takes a Poisson model simulation as argument) over multiple simulations
    Args:

        target_funcs: List[function] - functions to be computed
        nSims: int - number of simulations to average over
        target_dims: List[int] - output dimensionality of target_func
        base_model: rateMod_sim_fast - initialised Poisson model with pre-computed lookup table for the transfer function
        warning_spec: string - message to be displayed in warning when error is above 5%

    Returns: List[float or Tuple[float]] - List with average results for all target functions
    Out: array with dim par1xpar2xtarget_dim
    """
    if type(funcs) is not list:
        funcs = [funcs]
        target_dims = [target_dims]
    res = [np.zeros([len(vals1), len(vals2), target_dims[i]]) for i in range(len(funcs))]
    for i, par_i in tqdm(enumerate(vals1)):
        if par1 == 'x':
            simArgs['input_fct'] = lambda t: input_cst(t, par_i)
        else:
            exec(f"simArgs['{par1}']={par_i}")
        for j, par_j in enumerate(vals2):
            exec(f"simArgs['{par2}']={par_j}")
            spec_str = f' for {par1}: {par_i}, {par2}: {par_j}'
            res_raw = avs_over_sims_p(funcs, nSims,
                                      target_dims=target_dims, base_model=base_model,
                                      warning_spec=spec_str, **simArgs)
            for k, res_k in enumerate(res_raw):
                res[k][i, j, :] = res_k

    if len(res) == 1:
        return np.squeeze(res[0])
    else:
        return [np.squeeze(res_i) for res_i in res]


# Convenience functions
def fr_over_tau(tau_vals, nSims=10, base_model=None, **simArgs):
    """
    Compute firing rates of Poisson model for varying time-scales.
    Args:
        tau_vals: List[float] - values of time-scle
        nSims: int - number of simulations to evaluate over
        base_model: rateMod_sim_fast - optional: Poisson simulation instance with pre-computed lookup table for LIF transfer function
        **simArgs: other simulation arguments

    Returns: List[np.ndarray, np.ndarray] - results for firing rates of both plus and minus neurons

    """
    res = av_fct_over_1Dgrid_p(mean_fir_rates, 'tau', tau_vals, nSims,
                               target_dim=2, base_model=base_model)
    fr_plus, fr_minus = np.hsplit(res, 2)
    return fr_plus, fr_minus


def cv_over_tau(tau_vals, nSims=10, base_model=None, **simArgs):
    """
    Compute coefficient of variation of Poisson model for varying time-scales.
    Args:
        tau_vals: List[float] - values of time-scle
        nSims: int - number of simulations to evaluate over
        base_model: rateMod_sim_fast - optional: Poisson simulation instance with pre-computed lookup table for LIF transfer function
        **simArgs: other simulation arguments

    Returns: List[np.ndarray] - results for each given value of tau_h

    """
    res = av_fct_over_1Dgrid_p(cv, 'tau', tau_vals, nSims, base_model=base_model)
    return res
