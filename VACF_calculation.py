# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:50:25 2024

@author: Student
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.integrate
import pandas as pd


K = 0.15*10**-3
m_f = 0
shear = 0.0009

k_b = scipy.constants.k
T = 293
a = 1.5*10**-6
density = 2500
m = 4/3*np.pi*a**3*density
tau_f = density * a**2 / shear
gamma_s =  6 *np.pi*shear*a

integ_points = 10**4*8

power = np.linspace(-8, -4, 40)
times = (np.ones(len(power))*10)**power


def incompressible_admittance(omega):
    return 1 / (-1j*omega*(m + m_f/2) + gamma_s*(1+np.sqrt(-1j*omega*tau_f)) + K / (-1j*omega))

def incompressible_gamma(omega):
    return gamma_s*(1 + np.sqrt(-1j*omega*tau_f))

def velocity_spectral_density(omega, admit_func):
    return 2*k_b*T*np.real(admit_func(omega))

def position_spectral_density(omega, admit_func):
    return velocity_spectral_density(omega, admit_func) / omega**2


def thermal_force_PSD(omega, SPD, gamma, mass):
    G = (-1*omega**2*mass - 1j*omega*gamma + K)**-1
    return np.abs(G)**-2*SPD


def ACF_from_SPD(admit_function,SPD_func, times):
    low_freq = np.linspace(1, 10**4, integ_points)
    mid_freq = np.linspace(10**4, 10**6, integ_points)
    high_freq = np.linspace(10**6, 10**9, integ_points)
    top_freq = np.linspace(10**9, 10**12, integ_points)

    frequencies = np.concatenate((low_freq, mid_freq, high_freq, top_freq))
    ACF = np.zeros(len(times))

    for i in range(len(times)):
        ACF[i] = 2*np.real(scipy.integrate.simps(SPD_func(frequencies, admit_function)*np.exp(-1j*frequencies*times[i]), frequencies))/(2*np.pi)

    
    return ACF


def get_VACF(ms, aa, times):
    global m
    global a
    global tau_f
    global gamma_s
    m = ms
    a = aa
    tau_f = density * a**2 / shear
    gamma_s =  6 *np.pi*shear*a
    VACF_incompressible = ACF_from_SPD(incompressible_admittance,velocity_spectral_density, times) 
    return times, VACF_incompressible



times, one_micron_VACF = get_VACF(m, 0.5*10**-6, times)
times, three_micron_VACF = get_VACF(m, 1.5*10**-6, times)
times, three_1_micron_VACF = get_VACF(m, 1.55*10**-6, times)



plt.plot(times, one_micron_VACF, label = "1 micron")
plt.plot(times, three_micron_VACF, label = "3 microns")
plt.plot(times, three_1_micron_VACF, label = "3.1 microns")
plt.xscale("log")
plt.legend()





