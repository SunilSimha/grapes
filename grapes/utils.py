"""
Utils for handling grape profiles and related computations.
"""

from grapes.defs import cosmo
import numpy as np
from scipy.interpolate import (
    CubicSpline,
    PchipInterpolator,
    Akima1DInterpolator,
    CubicHermiteSpline,
    BarycentricInterpolator,
    KroghInterpolator,
)

# FloaterHormannInterpolator was added in scipy 1.13.0
try:
    from scipy.interpolate import FloaterHormannInterpolator
    HAS_FLOATER_HORMANN = True
except ImportError:
    HAS_FLOATER_HORMANN = False


def _create_fb_interpolators_general(table, index_col, radius_col, fb_col, 
                                     radius_scale=1.0, fb_scale=1.0, 
                                     synthetic_multipliers=None, additional_mask=None,
                                     interpolator_type='PchipInterpolator', spline_args=None):
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
        Multipliers for creating synthetic points beyond data range (default: np.linspace(1.8, 10))
    
    additional_mask : callable
        Optional function that takes table and index_value and returns additional boolean mask to apply
    
    interpolator_type : str
        Type of interpolator to use (default: 'PchipInterpolator')
        Options: 'CubicSpline', 'PchipInterpolator', 'Akima1DInterpolator',
                 'CubicHermiteSpline', 'BarycentricInterpolator', 
                 'KroghInterpolator', 'FloaterHormannInterpolator'
    
    spline_args : dict or None
        Additional keyword arguments to pass to the interpolator constructor (default: None)
    
    Returns:
    --------
    dict : Dictionary mapping index values to interpolator objects of the specified type
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
        
        if spline_args is None:
            spline_args = {}

        # Create interpolator based on specified type
        if interpolator_type == 'CubicSpline':
            # CubicSpline supports explicit boundary conditions
            interp_kwargs = {'bc_type': ((1, slope_left), (1, slope_right)), 'extrapolate': True}
            interp_kwargs.update(spline_args)
            f_b_interpolators[idx] = CubicSpline(radius_extended, fb_extended, **interp_kwargs)
        
        elif interpolator_type == 'CubicHermiteSpline':
            # CubicHermiteSpline requires explicit derivatives at all points
            slopes = np.gradient(fb_extended, radius_extended)
            slopes[0] = slope_left
            slopes[-1] = slope_right
            interp_kwargs = {'extrapolate': True}
            interp_kwargs.update(spline_args)
            f_b_interpolators[idx] = CubicHermiteSpline(
                radius_extended, fb_extended, dydx=slopes, **interp_kwargs
            )
        
        elif interpolator_type in ['PchipInterpolator', 'Akima1DInterpolator', 
                                    'BarycentricInterpolator', 'KroghInterpolator',
                                    'FloaterHormannInterpolator']:
            # These interpolators don't have explicit boundary condition support,
            # so we add anchor points to enforce endpoint slopes
            if interpolator_type == 'FloaterHormannInterpolator' and not HAS_FLOATER_HORMANN:
                raise ImportError(
                    "FloaterHormannInterpolator requires scipy >= 1.13.0. "
                    "Please upgrade scipy or choose a different interpolator."
                )
            
            if len(radius_extended) > 1:
                dx_left = radius_extended[1] - radius_extended[0]
                dx_right = radius_extended[-1] - radius_extended[-2]
            else:
                dx_left = dx_right = max(radius_extended[0] * 1e-3, 1e-6)

            left_x = radius_extended[0] - dx_left
            left_y = fb_extended[0] - slope_left * dx_left
            right_x = radius_extended[-1] + dx_right
            right_y = fb_extended[-1] + slope_right * dx_right

            radius_augmented = np.concatenate([[left_x], radius_extended, [right_x]])
            fb_augmented = np.concatenate([[left_y], fb_extended, [right_y]])
            
            # Build interpolator based on type
            if interpolator_type == 'PchipInterpolator':
                interp_kwargs = {'extrapolate': True}
                interp_kwargs.update(spline_args)
                f_b_interpolators[idx] = PchipInterpolator(
                    radius_augmented, fb_augmented, **interp_kwargs
                )
            elif interpolator_type == 'Akima1DInterpolator':
                interp_kwargs = {'extrapolate': True, 'method': 'makima'}
                interp_kwargs.update(spline_args)
                f_b_interpolators[idx] = Akima1DInterpolator(
                    radius_augmented, fb_augmented, **interp_kwargs
                )
            elif interpolator_type == 'BarycentricInterpolator':
                # BarycentricInterpolator doesn't support extrapolate parameter
                f_b_interpolators[idx] = BarycentricInterpolator(
                    radius_augmented, fb_augmented, **spline_args
                )
            elif interpolator_type == 'KroghInterpolator':
                # KroghInterpolator doesn't support extrapolate parameter
                f_b_interpolators[idx] = KroghInterpolator(
                    radius_augmented, fb_augmented, **spline_args
                )
            elif interpolator_type == 'FloaterHormannInterpolator':
                # FloaterHormannInterpolator doesn't support extrapolate parameter
                f_b_interpolators[idx] = FloaterHormannInterpolator(
                    radius_augmented, fb_augmented, **spline_args
                )
        else:
            raise ValueError(
                f"Unknown interpolator_type: {interpolator_type}. "
                f"Supported types: 'CubicSpline', 'PchipInterpolator', 'Akima1DInterpolator', "
                f"'CubicHermiteSpline', 'BarycentricInterpolator', 'KroghInterpolator', "
                f"'FloaterHormannInterpolator'"
            )
    
    return f_b_interpolators


def create_crocodile_interpolators(table, radius_scale=1.0, AGN_label="f",
                                  interpolator_type='PchipInterpolator', spline_args=None):
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
    
    interpolator_type : str
        Type of interpolator to use (default: 'PchipInterpolator')
    
    spline_args : dict or None
        Additional keyword arguments to pass to the interpolator constructor
    
    Returns:
    --------
    dict : Dictionary mapping logM_lo values to interpolator objects
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
        additional_mask=agn_mask,
        interpolator_type=interpolator_type,
        spline_args=spline_args
    )


def create_A23_interpolators(table, radius_scale=1.0, 
                            interpolator_type='PchipInterpolator', spline_args=None):
    """
    Create interpolation functions for f_b from Ayromlou+2023 simulation data.
    
    Parameters:
    -----------
    table : astropy.table.Table
        Table containing Ayromlou+2023 data with columns 'halo_mass_index', 'x=R/R200c', and 'y=f_b(<R)/f_b,cosmic'
    radius_scale : float
        Scaling factor for radius (default is 1.0, which means R/R200c as input to interpolator)
    
    interpolator_type : str
        Type of interpolator to use (default: 'PchipInterpolator')
    
    spline_args : dict or None
        Additional keyword arguments to pass to the interpolator constructor
    
    Returns:
    --------
    dict : Dictionary mapping halo_mass_index values to interpolator objects
    """
    return _create_fb_interpolators_general(
        table,
        index_col='halo_mass_index',
        radius_col='x=R/R200c',
        radius_scale=radius_scale,
        fb_col='y=f_b(<R)/f_b,cosmic',
        fb_scale=cosmo.Ob0 / cosmo.Om0,
        interpolator_type=interpolator_type,
        spline_args=spline_args
    )