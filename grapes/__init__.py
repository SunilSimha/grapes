"""
GRAPES: Generalized Radial Aggregated Profile Estimator for Simulations

A Python package for modeling baryon density profiles in dark matter halos
using the GRAPE framework. GRAPE provides a self-consistent method to derive
baryon density profiles from observed or simulated baryon fraction profiles,
ensuring proper asymptotic behavior and respecting cosmological constraints.

Key Features
------------
- NFW dark matter profiles with customizable concentration
- GRAPE baryon profiles from arbitrary f_b(r) functions
- Support for simulation data (CROCODILE, Ayromlou+2023)
- Diagnostic plotting capabilities
- Flexible interpolation utilities

Example Usage
-------------
>>> from grapes import GrapeNFWProfile
>>> import numpy as np
>>> 
>>> # Define a baryon fraction function
>>> def my_fb_func(r):
...     return 0.05 * np.tanh(r / 100.0) + 0.15
>>> 
>>> # Create a GRAPE profile
>>> profile = GrapeNFWProfile(log_M_halo_dm=12.0, f_b_func=my_fb_func)
>>> 
>>> # Compute density at various radii
>>> r_values = np.logspace(0, 3, 100)  # pc
>>> rho_b = profile.density(r_values)
>>> 
>>> # Generate diagnostic plots
>>> profile.diagnostic_plots(output_dir='./plots')
"""

__version__ = '0.1.0'
__author__ = 'Sunil Simha'
__license__ = 'MIT'
__url__ = 'https://github.com/SunilSimha/grapes'

# Core profile classes - primary user-facing API
from .grapes import (
    RadialProfile,
    NFWProfile,
    GrapeNFWProfile,
)

# Utility functions for working with simulation data
from .utils import (
    create_crocodile_interpolators,
    create_A23_interpolators,
)

# Default cosmology (can be imported if users need to customize)
from .defs import cosmo

# Define public API - controls "from grapes import *"
__all__ = [
    # Core classes
    'RadialProfile',
    'NFWProfile',
    'GrapeNFWProfile',
    # Utility functions
    'create_crocodile_interpolators',
    'create_A23_interpolators',
    # Cosmology
    'cosmo',
    # Package metadata
    '__version__',
    '__author__',
    '__license__',
    '__url__',
]


# Convenience function for version checking
def get_version():
    """
    Return the package version string.
    
    Returns
    -------
    str
        Version string in format 'major.minor.patch'
    """
    return __version__


def get_info():
    """
    Print package information.
    
    Displays version, author, license, and URL information.
    """
    print(f"GRAPES v{__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print(f"URL: {__url__}")
    print("\nGeneralized Radial Aggregated Profile Estimator for Simulations")
    print("For documentation, visit the GitHub repository.")
