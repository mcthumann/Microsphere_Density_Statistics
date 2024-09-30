import numpy as np

def sample_radius_density(radius_mean, radius_std_dev, density_mean, density_std_dev):
    """
    Samples a radius value and a density value from two normal distributions.

    Returns:
    tuple: Sampled radius and density values.
    """
    radius = np.random.normal(radius_mean, radius_std_dev)
    density = np.random.normal(density_mean, density_std_dev)

    return radius, density


if __name__ == "__main__":
    # Define means and variances for the normal distributions
    radius_mean = 3e-6
    radius_uncertainty = .03  # Quoted percent uncertainty
    density_mean = 2200  # Mean of the density distribution
    density_uncertainty = .03  # Quoted percent uncertainty

    radius_std_dev = radius_uncertainty*radius_mean
    density_std_dev = density_uncertainty*density_mean
    # Sample radius and density
    radius, density = sample_radius_density(radius_mean, radius_std_dev, density_mean, density_std_dev)

    # Output the sampled values
    print(f"Sampled radius: {radius}")
    print(f"Sampled density: {density}")
