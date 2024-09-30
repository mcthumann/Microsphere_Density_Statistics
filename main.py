import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy
import math

# Global Parameters
class Const:
    eta = 1e-3 # Viscosity of water
    rho_f = 1000  # Density of water
    K = 100e-6
    T = 293
    k_b = scipy.constants.k

def get_VACF_from_VPSD(times, full_mass, gamma_s, tau_f, integ_points):
    frequencies = np.logspace(0, 12, integ_points)
    ACF = np.zeros(len(times))

    for i in range(len(times)):
        ACF[i] = 2 * np.real(
            scipy.integrate.simps(velocity_spectral_density(frequencies, full_mass, gamma_s, tau_f) * np.exp(-1j * frequencies * times[i]),
                                  frequencies)) / (2 * np.pi)

    return ACF

def incompressible_admittance(omega, full_mass, gamma_s, tau_f):
    return 1 / (-1j * omega * (full_mass) + gamma_s * (1 + np.sqrt(-1j * omega * tau_f)) + Const.K / (-1j * omega))

def velocity_spectral_density(omega, full_mass, gamma_s, tau_f):
    return 2 * Const.k_b * Const.T * np.real(incompressible_admittance(omega, full_mass, gamma_s, tau_f))

def sample_radius_density(radius_mean, radius_std_dev, density_mean, density_std_dev):
    """
    Samples a radius value and a density value from two normal distributions.

    Returns:
    tuple: Sampled radius and density values.
    """
    radius = np.random.normal(radius_mean, radius_std_dev)
    density = np.random.normal(density_mean, density_std_dev)

    print("Sampled Radius is " + str(radius))
    print("Sampled Density is " + str(density))

    return radius, density

def fit_to_ACF(times, measured_vacf, radius_mean, gamma_s, tau_f, integ_points):
    # Initial guess for the density, can be the mean density
    density_guess = 2200

    # Define the fitting function where we only vary the density
    def vacf_fit_density(times, density):
        mass = (4 / 3) * math.pi * (radius_mean) ** 3 * density
        full_mass = mass + .5 * (4 / 3) * math.pi * (radius_mean) ** 3 * Const.rho_f  # Mass plus added mass
        return get_VACF_from_VPSD(times, full_mass, gamma_s, tau_f, integ_points)

    # Fit the VACF by adjusting the density only
    popt, pcov = curve_fit(vacf_fit_density, times, measured_vacf, p0=[density_guess])

    return popt[0]  # Return the best-fit density

if __name__ == "__main__":
    # Define means and variances for the normal distributions
    radius_mean = 1.5e-6
    radius_uncertainty = .03  # Quoted percent uncertainty
    density_mean = 2200  # Mean of the density distribution
    density_uncertainty = .03  # Quoted percent uncertainty

    radius_std_dev = radius_uncertainty*radius_mean
    density_std_dev = density_uncertainty*density_mean

    fit_densities = []
    for i in range(10):
        # Sample radius and density
        radius, density = sample_radius_density(radius_mean, radius_std_dev, density_mean, density_std_dev)

        times = np.logspace(-8, -5, 60)
        integ_points = 80000

        mass = (4 / 3) * math.pi * (radius) ** 3 * density
        full_mass = mass + .5 * (4 / 3) * math.pi * (radius) ** 3 * Const.rho_f  # Mass plus added mass
        gamma_s = 6 * math.pi * radius * Const.eta
        tau_f = Const.rho_f * radius ** 2 / Const.eta

        vacf = get_VACF_from_VPSD(times, full_mass, gamma_s, tau_f, integ_points)
        # plt.plot(times, vacf, label= f"Rho {density:.2e}kg/m^3, Mass {full_mass:.2e}kg".replace('e', ' × 10^'))
        # plt.xscale("log")
        # plt.show()

        # vacf = get_VACF_from_simulation()

        est_density = fit_to_ACF(times, vacf, radius_mean, gamma_s, tau_f, integ_points)
        fit_densities.append(est_density)

        print("Estimated Density is "+ str(est_density))

    # Plot the rolling mean for fit_densites
    rolling_mean = [np.mean(fit_densities[:i+1]) for i in range(len(fit_densities))]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_mean, label='Rolling Mean', marker='o')
    plt.axhline(density_mean, color='r', linestyle='--', label=f'Real Mean = {density_mean}')
    plt.title('Rolling Mean vs Real Mean')
    plt.xlabel('Index')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.grid(True)
    plt.show()
