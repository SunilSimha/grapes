"""
Utils for handling grape profiles and related computations.
"""

from grapes.defs import cosmo
import numpy as np
from scipy.interpolate import CubicSpline

def create_crocodile_interpolators(table, AGN_label = "f"):
    """
    Create interpolation functions for f_b from CROCODILE simulation data.
    
    Parameters:
    -----------
    table : astropy.table.Table
        Table containing CROCODILE data with columns logM_lo, R_over_R200, fb_med_norm
    
    AGN_label : str
        Label for the AGN type to select in the table (default is "f" for "fiducial")
        Allowed values are "f", and "n".
    
    Returns:
    --------
    dict : Dictionary mapping logM_lo values to CubicSpline interpolators
    """
    f_b_interpolators = {}
    
    # Get unique halo mass bins
    unique_logM = np.unique(table['logM_lo'])
    
    for logM in unique_logM:
        # Extract data for this halo mass bin
        mask = (table['logM_lo'] == logM) & (table['label_AGN'] == AGN_label)
        R_data = np.array(table['R_over_R200'][mask])
        fb_data = np.array(table['fb_med_norm'][mask])
        
        # Convert normalized f_b to actual f_b
        # fb_med_norm is f_b / f_b_cosmic, so multiply by cosmic baryon fraction
        fb_data = fb_data * cosmo.Ob0 / cosmo.Om0
        
        if len(R_data) < 3:
            print(f'Skipping logM={logM:.1f} due to insufficient data points')
            continue
        
        # Sort data
        sort_idx = np.argsort(R_data)
        R_data = R_data[sort_idx]
        fb_data = fb_data[sort_idx]
        
        # Estimate slopes for boundary conditions
        slope_left = (fb_data[1] - fb_data[0]) / (R_data[1] - R_data[0])
        slope_right = 0.0  # Asymptote to constant at large radius
        
        # Create CubicSpline with boundary conditions
        f_b_interpolators[logM] = CubicSpline(R_data, fb_data,
                                              bc_type=((1, slope_left), (1, slope_right)),
                                              extrapolate=True)
    
    return f_b_interpolators

def create_A23_interpolators(table):
    """
    Create interpolation functions for f_b from CROCODILE simulation data.
    Parameters:
    -----------
    table : astropy.table.Table
        Table containing CROCODILE data with columns 'halo_mass_index', 'x=R/R200c', and 'y=f_b(<R)/f_b,cosmic'
    Returns:
    --------
    dict : Dictionary mapping halo_mass_index values to CubicSpline interpolators
    """
    # Dictionary to store interpolating functions for each halo_mass_index
    f_b_interpolators = {}

    # Get unique halo mass indices from the table
    unique_halo_mass_indices = np.unique(table['halo_mass_index'])

    for idx in unique_halo_mass_indices:  # loop over halo mass indices
        mask = table['halo_mass_index'] == idx
        y_data = np.array(table['x=R/R200c'][mask]) * 1
        f_data = np.array(table['y=f_b(<R)/f_b,cosmic'][mask]) * cosmo.Ob0 / cosmo.Om0
        
        if len(y_data) < 3:
            print(f'Skipping interpolation for halo_mass_index={idx} due to insufficient data points')
            continue
        
        # Sort data by y_data to ensure proper ordering
        sort_idx = np.argsort(y_data)
        y_data = y_data[sort_idx]
        f_data = f_data[sort_idx]
        
        # Estimate the slope at the smallest radius (left boundary)
        # Using finite difference between first two points
        slope_left = (f_data[1] - f_data[0]) / (y_data[1] - y_data[0])
        
        # Set right boundary derivative to 0 (as y → inf)
        slope_right = 0.0
        
        # Create a CubicSpline with specified boundary conditions
        # bc_type allows us to set first derivatives at boundaries
        # bc_type format: ((order_at_left, value_at_left), (order_at_right, value_at_right))
        # order=1 means first derivative
        f_b_interpolators[idx] = CubicSpline(y_data, f_data, 
                                            bc_type=((1, slope_left), (1, slope_right)), extrapolate=True)

    return f_b_interpolators