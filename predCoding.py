#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:56:47 2021

@author: spokoiny
"""

from random import choice

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import erfcx


def lif_transferFct(mu, lambd, sigma_V, Vthr, Vreset, Vrest=0):
    """
    Steady-state solution of SDE
    dV/dt=-lambd*V+mu+sqrt(2*lambd*sigma_V)*eta,
    where eta is white noise
    """
    noiseScale = np.sqrt(2) * sigma_V
    a1 = (mu / lambd - Vthr) / noiseScale
    a2 = (mu / lambd - Vreset) / noiseScale
    T, _ = quad(func=erfcx, a=a1, b=a2)
    T = T * np.sqrt(np.pi)
    return lambd / T


lif_transferFct = np.vectorize(lif_transferFct)


def randomInit(model, sparse):
    """
    Random initialisation of membrane potential to prevent excessive synchronisation
    Args:
        model: str - if 'poisson', the magnitude is adjusted to correspond to the input potential h
        sparse: bool - see main simulation
    Returns: None

    """
    nNeur = model.nNeur
    if sparse:
        model.Vmembr = (2 * np.random.random(nNeur) - 1) * model.Vthr / 2
    else:
        model.Vmembr[:, 0] = (2 * np.random.random(nNeur) - 1) * model.Vthr / 2
    if 'pois' in model.type:
        model.Vmembr = model.Vmembr * model.lambd


class pc_model:
    """
    Class to represent a general predictive coding model
    """
    def __init__(self, nNeur=100, ratePosWeights=0.5, lambd=0.05, sigma_V=0.4, linCost=1,
                 quadCost=1, connStrength=5, tau=None):
        """

        Args:
            nNeur: int - number of neurons
            ratePosWeights: float - fraction of neurons with positive weight (in one-dimensional case)
            lambd: float - inverse of membrane time constant
            sigma_V: float - noise intensity
            linCost: float - linear cost parameter
            quadCost: float - quadratic cost parameter
            connStrength: float - strength of connection J for binary weights
            tau: None or float - time-scale for Poisson model (None for LIF model)
        """
        self.nNeur = nNeur
        self.ratePosWeights = ratePosWeights
        self.lambd = lambd
        self.sigma_V = sigma_V
        self.linCost = linCost
        self.quadCost = quadCost
        self.connStrength = connStrength
        self.noise_lvl = np.sqrt(2 * lambd) * sigma_V
        self.tau = tau

    def gen_sim_setup(self, input_fct=lambda t: 0 * t, T=200, t_step=0.01, balance='loose',
                      randInit=True, sparse=False):
        """
        General simulation setup (i.e. main definitions and initialisation) for both LIF and Poisson simulations
        Args:
            input_fct: function - input signal s
            T: float - simulation time
            t_step: float - time step for euler method
            balance: string or False - options are 'loose', 'classical' or False (no scaling)
            randInit: bool - whether to use random initialisation
            sparse: bool - whether to store membrane potentials over time (if True, also saves memory in general)

        Returns: tuple[np.ndarray,np.ndarray] - Tuple with two elements:
        noise for the simulation (None if sparse=True) and initialisation of membrane potential derivative

        """
        self.T = T  # Total duration in ms
        self.t_step = t_step  # for discretized time grid
        self.t_grid = np.arange(0, T + t_step, t_step)
        t_grid = self.t_grid  # time grid
        self.nSteps = len(t_grid)
        nSteps = self.nSteps  # this includes 'step' 0
        self.input_fct = input_fct
        self.input_s = input_fct(t_grid)
        if self.input_s.ndim == 1:  # for 1D signals
            self.input_s = self.input_s[np.newaxis]
        self.nDim = self.input_s.shape[0]
        nDim = self.nDim
        nNeur = self.nNeur
        nPosWeights = int(np.ceil(self.ratePosWeights * nNeur))
        patterns = np.ones(nNeur * nDim, dtype=int)
        patterns[nPosWeights:] = -1
        # Jconns=np.random.choice(Jconns,nNeur*nDim,replace=False)
        self.patterns = patterns.reshape([nDim, nNeur])
        patterns = self.patterns
        self.Jconns = self.connStrength * patterns
        # scaling rules
        if balance == 'loose':
            b_N = self.connStrength / nNeur
            a_N = nNeur
        elif balance == 'classical':
            b_N = self.connStrength / nNeur
            a_N = nNeur*np.sqrt(nNeur)
        else:
            b_N = self.connStrength
            a_N = 1
        self.recurr_scalFactor = a_N * b_N ** 2
        self.extInput_scalFactor = a_N * b_N
        self.weights = b_N * patterns
        self.Vthr = 1 / 2 * (
                a_N * b_N ** 2 * nDim + self.linCost + self.quadCost)

        self.Vreset = self.Vthr - self.quadCost - a_N * b_N ** 2 * nDim
        # initialize target variables
        self.signal = np.zeros([nDim, nSteps])
        self.predSignal = np.zeros([nDim, nSteps])
        if sparse:
            self.extInput = np.zeros(nNeur)
            self.Vmembr = np.zeros(nNeur)
            Vmembr_dt = np.zeros(nNeur)
            self.spikes = np.zeros([nNeur, nSteps], dtype=bool)
            self.firingRates = np.zeros(nNeur)
            output_list = [None, Vmembr_dt]
        else:
            noise = self.noise_lvl * np.random.randn(nNeur, self.nSteps)
            self.extInput = np.zeros([nNeur, nSteps])
            self.Vmembr = np.zeros([nNeur, nSteps])
            Vmembr_dt = np.zeros(nNeur)
            self.spikes = np.zeros([nNeur, nSteps])
            self.firingRates = np.zeros([nNeur, nSteps])
            output_list = [noise, Vmembr_dt]
        if randInit: randomInit(self, sparse)
        return (output_list)

    def lif_transferFct(self, mu):  # cleaner def? xxx
        return lif_transferFct(mu, self.lambd, self.sigma_V, self.Vthr, self.Vreset)


class gen_model(pc_model):
    """
    Class to obtain general network parameters without performing specific computations
    """
    def __init__(self, input_fct=lambda t: 0 * t, T=200, t_step=0.01, nNeur=100,
                 ratePosWeights=0.5, lambd=0.05, sigma_V=0.4, linCost=1,
                 quadCost=1, connStrength=5,balance='loose', tau=None, randInit=True):
        randn = np.random.randn
        self.type = 'gen'
        super().__init__(nNeur, ratePosWeights, lambd, sigma_V, linCost,
                         quadCost, connStrength, tau)
        _ = super().gen_sim_setup(input_fct, T, t_step, balance=balance, randInit=randInit, sparse=True)


class spikeMod_sim(pc_model):
    """
    LIF simulation
    """
    def __init__(self, input_fct=lambda t: 0 * t, T=200, t_step=0.01, nNeur=100,
                 ratePosWeights=0.5, lambd=0.05, sigma_V=0.4, linCost=1,
                 quadCost=1, connStrength=5, balance='loose', tau=None, randInit=True, sparse=False,
                 asynchr=False):
        """
        Args:
            input_fct: function - input signal s
            T: float - simulation time
            t_step: float - time step for euler method
            nNeur: number of neurons
            ratePosWeights: float - fraction of neurons with positive weight (in one-dimensional case)
            lambd: float - inverse of membrane time constant
            sigma_V: float - noise intensity
            linCost: float - linear cost parameter
            quadCost: float - quadratic cost parameter
            connStrength: float - strength of connection J for binary weights
            balance: string or False - options are 'loose', 'classical' or False (no scaling)
            tau: None or float - time-scale for Poisson model (None for LIF model)
            randInit: bool - whether to use random initialisation
            sparse: bool - whether to store membrane potentials over time (if True, also saves memory in general)
            asynchr: bool - if True, only ever allow one spike per time step
        """
        self.type = 'lif_model'
        randn = np.random.randn
        super().__init__(nNeur, ratePosWeights, lambd, sigma_V, linCost,
                         quadCost, connStrength)

        noise, Vmembr_dt = super().gen_sim_setup(input_fct, T, t_step, balance=balance, randInit=randInit,
                                                 sparse=sparse)  # xxx change this as for rateMod_fast
        Vmembr = self.Vmembr
        patterns = self.patterns
        for t in range(1, self.nSteps):
            self.signal[:, t] = self.signal[:, t - 1] + t_step * \
                                (-lambd * self.signal[:, t - 1] + self.input_s[:, t - 1])
            if sparse:
                self.extInput = self.extInput_scalFactor * patterns.T @ self.input_s[:, t]
                Vmembr_dt = -lambd * Vmembr + self.extInput
                noise_t = self.noise_lvl * randn(nNeur)
                Vmembr = Vmembr + t_step * Vmembr_dt + \
                         np.sqrt(t_step) * noise_t - self.recurr_scalFactor * (
                                 patterns.T @ (patterns @ self.spikes[:, t - 1])) - \
                         self.quadCost * self.spikes[:, t - 1]
                spikes_t = np.where(Vmembr >= self.Vthr)[0]

            else:
                self.extInput[:, t] = self.extInput_scalFactor * patterns.T @ self.input_s[:, t]
                Vmembr_dt = -lambd * Vmembr[:, t - 1] + self.extInput[:, t]
                Vmembr[:, t] = Vmembr[:, t - 1] + t_step * Vmembr_dt + \
                               np.sqrt(t_step) * noise[:, t] - self.recurr_scalFactor * (
                                           patterns.T @ (patterns @ self.spikes[:,
                                                                    t - 1])) - self.quadCost * self.spikes[:, t - 1]
                spikes_t = np.where(Vmembr[:, t] >= self.Vthr)[0]
                self.firingRates[:, t] = self.firingRates[:, t - 1] + t_step * \
                                         (-lambd * self.firingRates[:, t - 1]) + self.spikes[:, t]

            if asynchr and spikes_t.size > 1:
                # self.spikes_synchr[spikes_t,t]=1 xxx
                spikes_t = choice(spikes_t)
            self.spikes[spikes_t, t] = 1
            self.predSignal[:, t] = self.predSignal[:, t - 1] + t_step * \
                                    (-lambd * self.predSignal[:, t - 1]) + self.weights @ self.spikes[:, t]

        self.spike_neur, self.spike_times = np.where(self.spikes == 1)


class rateMod_sim(pc_model):
    """
    Poisson model simulation
    """
    def __init__(self, input_fct=lambda t: 0 * t, T=200, t_step=0.1, nNeur=100,
                 ratePosWeights=0.5, lambd=0.05, sigma_V=0.4, linCost=1,
                 quadCost=1, balance='loose', connStrength=5, tau=10):
        """
        Args:
            input_fct: function - input signal s
            T: float - simulation time
            t_step: float - time step for euler method
            nNeur: number of neurons
            ratePosWeights: float - fraction of neurons with positive weight (in one-dimensional case)
            lambd: float - inverse of membrane time constant
            sigma_V: float - noise intensity
            linCost: float - linear cost parameter
            quadCost: float - quadratic cost parameter
            balance: string or False - options are 'loose', 'classical' or False (no scaling)
            connStrength: float - strength of connection J for binary weights
            tau: None or float - time-scale for Poisson model (None for LIF model)
        """
        super().__init__(nNeur, ratePosWeights, lambd, sigma_V, linCost,
                         quadCost, connStrength, tau)
        self.type = 'poisson_model'
        noise, Vmembr_dt = super().gen_sim_setup(input_fct, T, t_step,balance=balance)
        unifSamples = np.random.rand(nNeur, self.nSteps)
        patterns=self.patterns
        for t in range(1, self.nSteps):
            self.signal[:, t] = self.signal[:, t - 1] + t_step * (
                    -lambd * self.signal[:, t - 1] + self.input_s[:, t - 1])
            self.extInput[:, t] = self.extInput_scalFactor * self.patterns.T @ self.input_s[:, t]
            Vmembr_dt[:, t] = -self.Vmembr[:, t - 1] + self.extInput[:, t]
            self.Vmembr[:, t] = self.Vmembr[:, t - 1] + t_step * Vmembr_dt[:, t] / tau - self.recurr_scalFactor/tau * \
                                (patterns.T @ (patterns @ self.spikes[:,t - 1]- spikes[:, t - 1]))
            self.firingRates[:, t] = lif_transferFct(self.Vmembr[:, t], lambd,
                                                     self.sigma_V, self.Vthr, self.Vreset)

            spikes_t = np.where(self.t_step * self.firingRates[:, t] > unifSamples[:, t])
            self.spikes[spikes_t, t] = 1
            self.predSignal[:, t] = self.predSignal[:, t - 1] + t_step * (
                    -lambd * self.predSignal[:, t - 1]) + self.weights @ self.spikes[:, t]

        self.spike_neur, self.spike_times = np.where(self.spikes == 1)


class rateMod_sim_fast(pc_model):
    def __init__(self, input_fct=lambda t: 0 * t, T=200, t_step=0.1, nNeur=100,
                 ratePosWeights=0.5, lambd=0.05, sigma_V=0.4, linCost=1,
                 quadCost=1, connStrength=5, balance='loose', tau=10, manualSim=False,
                 sparse=True, asynchr=False):
        """

        Args:
            input_fct: function - input signal s
            T: float - simulation time
            t_step: float - time step for euler method
            nNeur: number of neurons
            ratePosWeights: float - fraction of neurons with positive weight (in one-dimensional case)
            lambd: float - inverse of membrane time constant
            sigma_V: float - noise intensity
            linCost: float - linear cost parameter
            quadCost: float - quadratic cost parameter
            connStrength: float - strength of connection J for binary weights
            balance: string or False - options are 'loose', 'classical' or False (no scaling)
            tau: None or float - time-scale for Poisson model (None for LIF model)
            manualSim: if True, no simulation is performed (only general initialisation)
            sparse: bool - whether to store membrane potentials over time (if True, also saves memory in general)
            asynchr: bool - if True, only ever allow one spike per time step
        """
        super().__init__(nNeur, ratePosWeights, lambd, sigma_V, linCost,
                         quadCost, connStrength, tau)

        self.type = 'poisson_model'
        _ = super().gen_sim_setup(input_fct, T, t_step, balance=balance)
        # pre-calculated table of values for LIF transfer function
        self.lookup_knots, self.lif_lookup_table = self.calc_lif_lookup_table()
        if not manualSim:
            self.sim(balance=balance, sparse=sparse, asynchr=asynchr)

    def calc_lif_lookup_table(self, step_size=0.0001):
        """
        Compute Lookup table for LIF transfer function over a fine grid where the function is non-saturating
        Args:
            step_size: mesh size of interpolation grid

        Returns: List[np.ndarray,np.ndarray] - grid points and corresponding function values

        """
        upper_lim = minimize(lambda x: (1 - self.lif_transferFct(x)) ** 2, 0).x
        lower_lim = minimize(self.lif_transferFct, 0).x
        lookup_knots = np.arange(lower_lim, upper_lim + step_size, step_size)
        lif_lookup_table = self.lif_transferFct(lookup_knots)
        return [lookup_knots, lif_lookup_table]

    def interpolated_lif(self, h_input, knots, vals):
        """
        compute lif transfer function using interpolation over a lookup-table
        Args:
            h_input: input potential (argument)
            knots: interpolation knots
            vals: lookup values corresponding to the knots

        Returns: interpolation result

        """
        step_size = knots[1] - knots[0]
        nearest_lower_neighb = ((h_input - knots[0]) / step_size).astype(int)
        nearest_lower_neighb = np.clip(nearest_lower_neighb, 0, knots.size - 1)
        interpol_dist = np.maximum(h_input - knots[nearest_lower_neighb], 0)
        interpol_res = interpol_dist / step_size * vals[np.minimum(nearest_lower_neighb + 1, knots.size - 1)] + \
                       (1 - interpol_dist / step_size) * vals[nearest_lower_neighb]
        return interpol_res

    def sim(self, input_fct=None, tau=None, T=None, t_step=None,balance='loose', sparse=False, asynchr=False):
        """
        Simulate predictive coding network for given input. Allows to efficiently re-use pre-computed LIF lookup table
        and freely adjust parameters that do not affect the LIF transfer function

        Args:
            input_fct: function - input signal s
            tau: float - time-scale of Poisson model
            T: float - simulation time
            t_step: float - time step for euler method
            balance: str or False - options are 'loose', 'classical' or False (no scaling)
            sparse: bool - whether to store membrane potentials over time (if True, also saves memory in general)
            asynchr: bool - if True, only ever allow one spike per time step
        Returns: None

        """
        if input_fct == None:
            input_fct = self.input_fct
        else:
            self.input_fct = input_fct
        if T == None:
            T = self.T
        else:
            self.T = T
        if t_step == None:
            t_step = self.t_step
        else:
            self.t_step = t_step
        if tau == None:
            tau = self.tau
        else:
            self.tau = tau

        lookup_knots = self.lookup_knots
        lif_lookup_table = self.lif_lookup_table
        interpolated_lif = self.interpolated_lif
        _ = super().gen_sim_setup(input_fct, T, t_step, balance=balance, sparse=sparse, randInit=True)

        # for performance, define local variables
        rand = np.random.rand
        patterns, connStrength = self.patterns, self.connStrength
        t_step = self.t_step
        lambd = self.lambd
        input_s = self.input_s
        extInput = self.extInput
        signal = self.signal
        nNeur = self.nNeur
        weights = self.weights

        firingRates = self.firingRates
        predSignal = self.predSignal
        Vmembr = self.Vmembr
        spikes = self.spikes
        # xxx
        if asynchr: self.spikes_synchr = np.zeros([nNeur, self.nSteps], bool)
        if not sparse:
            unifSamples = rand(nNeur, self.nSteps)
        for t in range(1, self.nSteps):
            signal[:, t] = signal[:, t - 1] + t_step * (-lambd * signal[:, t - 1] + input_s[:, t - 1])
            if not sparse:
                extInput[:, t] = self.extInput_scalFactor * self.patterns.T @ self.input_s[:, t]
                # xxx here an above: include recurrent terms in Vmembr
                Vmembr_dt = -Vmembr[:, t - 1] + extInput[:, t]

                Vmembr[:, t] = Vmembr[:, t - 1] + t_step * Vmembr_dt / tau - self.recurr_scalFactor/tau * (
                        patterns.T @ (patterns @ self.spikes[:,t - 1]) - spikes[:, t - 1])
                firingRates[:, t] = interpolated_lif(Vmembr[:, t], lookup_knots, lif_lookup_table)
                spikes_t = np.where(t_step * firingRates[:, t] > unifSamples[:, t])[0]
            else:
                extInput = self.extInput_scalFactor * self.patterns.T @ self.input_s[:, t]
                # here and above: include recurrent terms in Vmembr
                Vmembr_dt = -Vmembr + extInput
                Vmembr = Vmembr + t_step * Vmembr_dt / tau - self.recurr_scalFactor/tau * (
                        patterns.T @ (patterns @ self.spikes[:,t - 1]) - spikes[:, t - 1])
                firingRates = interpolated_lif(Vmembr, lookup_knots,
                                               lif_lookup_table)
                spikes_t = np.where(t_step * firingRates > rand(nNeur))[0]
            if asynchr and spikes_t.size > 1:
                self.spikes_synchr[spikes_t, t] = 1
                spikes_t = choice(spikes_t)
            spikes[spikes_t, t] = 1
            predSignal[:, t] = predSignal[:, t - 1] + t_step * (-lambd) * predSignal[:, t - 1] + weights @ spikes[:, t]

        self.spikes = spikes
        self.spike_neur, self.spike_times = np.where(self.spikes == 1)
        self.extInput = extInput
        self.firingRates = firingRates
        self.Vmembr = Vmembr
        self.predSignal = predSignal