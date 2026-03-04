"""
Utils for handling grape profiles and related computations.
"""

from grapes.defs import cosmo
import numpy as np
from scipy.interpolate import CubicSpline


def _create_fb_interpolators_general(table, index_col, radius_col, fb_col, 
                                     radius_scale=1.0, fb_scale=1.0, 
                                     synthetic_multipliers=None, additional_mask=None):
    """
    General function to create baryon fraction interpolators from simulation data.
    
    Parameters:
    -----------
    table : astropy.table.Table
        Table containing the simulation data
    
    index_col : str
        Name of the column that identifies unique halo bins
    
    radius_col : str
        Name of the column containing radius values
    
    fb_col : str
        Name of the column containing baryon fraction values
    
    radius_scale : float
        Scaling factor for radius (default: 1.0)
    
    fb_scale : float or callable
        Scaling factor for f_b values. If callable, applied to fb_data (default: 1.0)
    
    synthetic_multipliers : array-like
        Multipliers for creating synthetic points beyond data range (default: [1.5, 2.0, 3.0, 5.0, 10.0])
    
    additional_mask : callable
        Optional function that takes table and index_value and returns additional boolean mask to apply
    
    Returns:
    --------
    dict : Dictionary mapping index values to CubicSpline interpolators
    """
    if synthetic_multipliers is None:
        synthetic_multipliers = np.linspace(1.8, 10)
    
    f_b_cosmic = cosmo.Ob0 / cosmo.Om0
    f_b_interpolators = {}
    
    # Get unique indices
    unique_indices = np.unique(table[index_col])
    
    for idx in unique_indices:
        # Base mask for this index
        mask = table[index_col] == idx
        
        # Apply additional mask if provided
        if additional_mask is not None:
            mask = mask & additional_mask(table, idx)
        
        # Extract and scale data
        radius_data = np.array(table[radius_col][mask]) * radius_scale
        fb_data = np.array(table[fb_col][mask])

        # Apply fb scaling
        if callable(fb_scale):
            fb_data = fb_scale(fb_data)
        else:
            fb_data = fb_data * fb_scale

        # Filter out non-finite values in radius and f_b
        finite_mask = np.isfinite(radius_data) & np.isfinite(fb_data)
        radius_data = radius_data[finite_mask]
        fb_data = fb_data[finite_mask]

        # Require at least three finite data points to build a reliable spline
        if len(radius_data) < 3:
            print(f'Skipping index={idx} due to insufficient finite data points')
            continue
        # Sort data
        sort_idx = np.argsort(radius_data)
        radius_data = radius_data[sort_idx]
        fb_data = fb_data[sort_idx]
        
        # Add synthetic points at large radii to asymptote to cosmic mean
        radius_max = radius_data[-1]
        radius_synthetic = radius_max * synthetic_multipliers
        fb_synthetic = np.full_like(radius_synthetic, f_b_cosmic)
        
        radius_extended = np.concatenate([radius_data, radius_synthetic])
        fb_extended = np.concatenate([fb_data, fb_synthetic])
        
        # Estimate slopes for boundary conditions
        slope_left = (fb_data[1] - fb_data[0]) / (radius_data[1] - radius_data[0])
        slope_right = 0.0  # Zero slope at large radius
        
        # Create CubicSpline with extended data and boundary conditions
        f_b_interpolators[idx] = CubicSpline(radius_extended, fb_extended,
                                             bc_type=((1, slope_left), (1, slope_right)),
                                             extrapolate=True)
    
    return f_b_interpolators


def create_crocodile_interpolators(table, radius_scale=1.0, AGN_label="f"):
    """
    Create interpolation functions for f_b from CROCODILE simulation data.
    
    Parameters:
    -----------
    table : astropy.table.Table
        Table containing CROCODILE data with columns logM_lo, R_over_R200, fb_med_norm
    radius_scale : float
        Scaling factor for radius (default is 1.0, which means R/R200 as input to interpolator)
    
    AGN_label : str
        Label for the AGN type to select in the table (default is "f" for "fiducial")
        Allowed values are "f", and "n".
    
    Returns:
    --------
    dict : Dictionary mapping logM_lo values to CubicSpline interpolators
    """
    def agn_mask(table, idx):
        return table['label_AGN'] == AGN_label
    
    return _create_fb_interpolators_general(
        table,
        index_col='logM_lo',
        radius_col='R_over_R200',
        fb_col='fb_med_norm',
        radius_scale=radius_scale,
        fb_scale=cosmo.Ob0 / cosmo.Om0,
        additional_mask=agn_mask
    )


def create_A23_interpolators(table, radius_scale=1.0):
    """
    Create interpolation functions for f_b from Ayromlou+2023 simulation data.
    
    Parameters:
    -----------
    table : astropy.table.Table
        Table containing Ayromlou+2023 data with columns 'halo_mass_index', 'x=R/R200c', and 'y=f_b(<R)/f_b,cosmic'
    radius_scale : float
        Scaling factor for radius (default is 1.0, which means R/R200c as input to interpolator)
    
    Returns:
    --------
    dict : Dictionary mapping halo_mass_index values to CubicSpline interpolators
    """
    return _create_fb_interpolators_general(
        table,
        index_col='halo_mass_index',
        radius_col='x=R/R200c',
        radius_scale=radius_scale,
        fb_col='y=f_b(<R)/f_b,cosmic',
        fb_scale=cosmo.Ob0 / cosmo.Om0
    )