# GRAPES

**Generalized Radial Aggregated Profile Estimator for Simulations**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python package for modeling baryon density profiles in dark matter halos using the GRAPE framework. GRAPE provides a self-consistent method to derive baryon density profiles from observed or simulated baryon fraction profiles, ensuring proper asymptotic behavior and respecting cosmological constraints.

## Overview

This repository implements the GRAPE (Generalized Radial Aggregated Profile Estimator) framework for describing baryon and dark matter density profiles in halos while ensuring physically motivated baryon to dark matter ratios. Given an arbitrary baryon fraction function $f_b(r)$, GRAPE derives the corresponding baryon density profile $\rho_b(r)$ that is physically consistent with the underlying dark matter distribution.

The method uses the relationship:

$$M_B(<r) = g(r) \cdot \frac{\Omega_B}{\Omega_M - \Omega_B} \cdot M_{DM}(<r)$$

where $g(r)$ is derived from the observed $f_b(r)$ profile, ensuring proper asymptotic behavior at large radii and self-consistency throughout the halo.

## Key Features

- **NFW Dark Matter Profiles**: Customizable concentration-mass relations
- **GRAPE Baryon Profiles**: Self-consistent baryon density profiles from arbitrary $f_b(r)$ functions
- **Simulation Data Support**: Built-in utilities for CROCODILE and Ayromlou+2023 simulation data
- **Column Density Calculations**: Projected column densities for observational comparisons
- **Diagnostic Plotting**: Comprehensive visualization tools for profile validation
- **Flexible Interpolation**: Utilities for working with tabulated simulation data

## Installation

### From Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/SunilSimha/grapes.git
cd grapes
pip install -e .
```

### Dependencies

GRAPES requires the following packages:

- `numpy >= 2.0.0`
- `scipy >= 1.11.0`
- `astropy >= 6.0.0`
- `matplotlib >= 3.8.0`

These will be automatically installed when you install GRAPES.

## Quick Start

### Basic Usage

```python
from grapes import GrapeNFWProfile
import numpy as np

# Define a baryon fraction function
def my_fb_func(r):
    """
    Baryon fraction as a function of radius (in pc).
    This example shows a smooth transition from low to high f_b.
    """
    return 0.05 * np.tanh(r / 100.0) + 0.15

# Create a GRAPE profile for a log(M_halo/Msun) = 12.0 halo
profile = GrapeNFWProfile(
    log_M_halo_dm=12.0,
    f_b_func=my_fb_func,
    concentration=7.67,
    redshift=0.0
)

# Compute baryon density at various radii
r_values = np.logspace(0, 3, 100)  # 1 to 1000 pc
rho_b = profile.density(r_values)  # Msun/pc^3

# Compute enclosed baryon mass
M_b_enclosed = profile.mass_enclosed(r_values)  # Msun

# Calculate column density at impact parameter
R_impact = 50  # pc
R_trunc = 500  # pc
N_H = profile.column_density(R_impact, R_trunc)  # integrated density along line of sight
```

### Using Simulation Data

GRAPES includes utilities for creating interpolators from simulation data:

```python
from grapes import GrapeNFWProfile, create_A23_interpolators, create_crocodile_interpolators
from astropy.table import Table

# Load simulation data (Ayromlou+2023 format)
sim_table = Table.read('path/to/simulation_data.fits')

# Create interpolators for different halo mass bins
fb_interpolators = create_A23_interpolators(sim_table)

# Use interpolator for a specific halo mass bin
log_M = 12.0
profile = GrapeNFWProfile(
    log_M_halo_dm=log_M,
    f_b_func=fb_interpolators[log_M],  # Use interpolated f_b(r) from simulation
    concentration=7.67
)
```

### Diagnostic Plots

Generate comprehensive diagnostic plots to validate your profiles:

```python
# Create diagnostic plots showing density, mass, and baryon fraction profiles
profile.diagnostic_plots(output_dir='./plots')
```

## Documentation

Detailed usage examples and theoretical background can be found in the Jupyter notebooks in the [`docs/`](docs/) directory:

- [`generalized_halo_profile.ipynb`](docs/generalized_halo_profile.ipynb): Theoretical derivation and mathematical framework
- [`grape_nfw_interpolated_profiles.ipynb`](docs/grape_nfw_interpolated_profiles.ipynb): Working with simulation data and creating GRAPE profiles
- [`utils_interpolators_demo.ipynb`](docs/utils_interpolators_demo.ipynb): Using interpolation utilities for CROCODILE and Ayromlou+2023 data

## Core Classes

### `RadialProfile`
Abstract base class for radial density profiles. Provides interface for `density()`, `mass_enclosed()`, and `column_density()` calculations.

### `NFWProfile`
Implements the Navarro-Frenk-White (NFW) profile for dark matter halos. Automatically computes scale radius and characteristic density from halo mass and concentration.

### `GrapeNFWProfile`
The main GRAPE implementation that derives baryon density profiles from user-specified baryon fraction functions. Ensures self-consistency with the underlying NFW dark matter distribution.

## Utilities

- `create_crocodile_interpolators()`: Create baryon fraction interpolators from CROCODILE simulation data
- `create_A23_interpolators()`: Create baryon fraction interpolators from Ayromlou+2023 simulation data

## Cosmology

GRAPES uses the Planck18 cosmology from `astropy` by default. The cosmology can be accessed as:

```python
from grapes import cosmo
print(cosmo.Om0, cosmo.Ob0, cosmo.H0)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GRAPES in your research, please cite:

```bibtex
@software{grapes2026,
  author = {Simha, Sunil},
  title = {GRAPES: Generalized Radial Aggregated Profile Estimator for Simulations},
  year = {2026},
  url = {https://github.com/SunilSimha/grapes},
  version = {0.1.0}
}
```

## Contact

- **Author**: Sunil Simha
- **Repository**: https://github.com/SunilSimha/grapes
- **Issues**: https://github.com/SunilSimha/grapes/issues

## Acknowledgments

This package builds on theoretical work in halo modeling and incorporates data from:
- CROCODILE simulations
- Ayromlou et al. 2023 (TNG simulations)
