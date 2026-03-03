"""
Class to handle generalized halo profiles.
"""

import numpy as np
from scipy.differentiate import derivative
from grapes.defs import cosmo

class RadialProfile:
    def __init__(self, r_s, rho_0, redshift=0.0, cosmology=cosmo):
        self.r_s = r_s
        self.rho_0 = rho_0
        self.redshift = redshift
        self.cosmology = cosmology

    def density(self, r):
        """Calculate the density at a given radius r."""
        pass

    def mass_enclosed(self, r):
        """Calculate the mass enclosed within radius r."""
        pass

    def column_density(self, impact_param, R_trunc, n_steps=1000):
        """Calculate the column density at projected radius R."""
        # This is a placeholder implementation. The actual integral should be computed.
        z_array = np.linspace(0, np.sqrt(R_trunc**2 - impact_param**2), n_steps)
        density_array = self.density(np.sqrt(impact_param**2 + z_array**2))
        return 2*np.trapezoid(density_array, z_array)


class NFWProfile(RadialProfile):

    def __init__(self, log_M_halo,
                 concentration=7.67,
                 redshift=0.0, cosmology=cosmo):
        
        self.cosmo = cosmology
        self.concentration = concentration
        self.M_halo = 10**log_M_halo
        q = self.cosmo.Ode0/(self.cosmo.Ode0+self.cosmo.Om0*(1+self.z)**3) 
        rho_crit = self.cosmo.critical_density(redshift).to_value('Msun/pc**3')
        r_vir = (3 * self.M_halo / (4 * np.pi * rho_crit * q))**(1/3)
        r_s = r_vir / concentration
        self.rho_0 = rho_crit * q / (concentration**2 * (1 + concentration)**2)
        super().__init__(r_s, self.rho_0)

    def density(self, r):
        """Calculate the NFW density at radius r."""
        x = r / self.r_s
        return self.rho_0 / (x * (1 + x)**2)

    def mass_enclosed(self, r):
        """Calculate the mass enclosed within radius r for NFW profile."""
        x = r / self.r_s
        return 4 * np.pi * self.rho_0 * self.r_s**3 * (np.log(1 + x) - x / (1 + x))
    


class GrapeNFWProfile(RadialProfile):
    """
    Generalized halo baryon profile using the GRAPE (Generalized Radial Aggregated Profile Estimator) framework.
    
    This class models baryon density profiles given a user-specified baryon fraction function f_b(r).
    It uses the relationship:
        M_B(<r) = g(r) * (Omega_B / (Omega_M - Omega_B)) * M_DM(<r)
    
    where g(r) is derived from the observed f_b(r) profile such that:
        g(r) = (Omega_M - Omega_B) / Omega_B * f_b(r) / (1 - f_b(r))
    
    The baryon density profile is then computed as:
        rho_B(r) = Omega_B/(Omega_M - Omega_B) * [g(r) * rho_DM(r) + (1/(4*pi*r^2)) * dg/dr * M_DM(<r)]
    """
    
    def __init__(self, log_M_halo_dm, f_b_func, concentration=7.67, 
                 redshift=0.0, cosmology=cosmo):
        """
        Initialize a GRAPE NFW baryon profile.
        
        Parameters
        ----------
        log_M_halo : float
            Log10 of the halo mass in solar masses
        f_b_func : callable
            Function that computes the baryon fraction f_b(r) at radius r (in physical or normalized units).
            Should accept array input and return array output.
            f_b should be defined such that 0 <= f_b(r) <= Omega_B/Omega_M
        concentration : float, optional
            NFW concentration parameter (default: 7.67)
        redshift : float, optional
            Redshift of the halo (default: 0.0)
        cosmology : astropy.cosmology.Cosmology, optional
            Cosmological model (default: Planck18)
        """
        self.f_b_func = f_b_func
        self.nfw_profile = NFWProfile(log_M_halo_dm, concentration=concentration, 
                                       redshift=redshift, cosmology=cosmology)
        
        # Initialize the RadialProfile with NFW parameters
        super().__init__(self.nfw_profile.r_s, self.nfw_profile.rho_0, 
                        redshift=redshift, cosmology=cosmology)
        
        # Store cosmological ratios for GRAPE calculations
        self.Omega_ratio = self.cosmo.Omega_B / (self.cosmo.Om0 - self.cosmo.Omega_B)
        
        self.redshift = redshift
        self.concentration = concentration
    
    def _compute_g(self, r):
        """
        Compute g(r) from f_b(r).
        
        g(r) = (Omega_M - Omega_B) / Omega_B * f_b(r) / (1 - f_b(r))
        
        Parameters
        ----------
        r : array-like
            Radius values
        
        Returns
        -------
        g : array-like
            Values of g(r)
        """
        f_b = self.f_b_func(r)
        # Avoid division by zero/infinity
        f_b = np.clip(f_b, 1e-10, 1 - 1e-10)
        return f_b / ((1.0 - f_b) * self.Omega_ratio)
    
    def _compute_dg_dr(self, r):
        """
        Compute the derivative dg/dr numerically.
        
        Parameters
        ----------
        r : array-like
            Radius values
        step : float, optional
            Step size for numerical differentiation (default: 1e-6)
        
        Returns
        -------
        dg_dr : array-like
            Values of dg/dr at each radius
        """
        def g_wrapper(r_val):
            return self._compute_g(r_val)
        
        dg_dr = derivative(g_wrapper, r).df
        return dg_dr
    
    def density(self, r):
        """
        Calculate the GRAPE baryon density profile at radius r.
        
        rho_B(r) = Omega_ratio * [g(r) * rho_DM(r) + (dg/dr * M_DM(<r)) / (4*pi*r^2)]
        
        Parameters
        ----------
        r : array-like
            Radius values
        
        Returns
        -------
        rho_B : array-like
            Baryon density at each radius
        """
        r = np.atleast_1d(r)
        
        # Get NFW dark matter profile components
        rho_dm = self.nfw_profile.density(r)
        M_dm = self.nfw_profile.mass_enclosed(r)
        
        # Compute g(r) and its derivative
        g = self._compute_g(r)
        dg_dr = self._compute_dg_dr(r)
        
        # Compute the two terms of the GRAPE baryon profile
        # Term 1: scaled dark matter profile
        term1 = g * rho_dm
        
        # Term 2: correction term ensuring proper enclosed mass ratio
        # Avoid division by zero at r=0
        with np.errstate(divide='ignore', invalid='ignore'):
            term2 = (dg_dr * M_dm) / (4 * np.pi * r**2)
            term2 = np.nan_to_num(term2, nan=0.0, posinf=0.0, neginf=0.0)
        
        return self.Omega_ratio * (term1 + term2)
    
    def mass_enclosed(self, r):
        """
        Calculate the enclosed baryon mass within radius r.
        
        M_B(<r) = g(r) * (Omega_B / (Omega_M - Omega_B)) * M_DM(<r)
        
        Parameters
        ----------
        r : array-like
            Radius values
        
        Returns
        -------
        M_B : array-like
            Enclosed baryon mass at each radius
        """
        r = np.atleast_1d(r)
        g = self._compute_g(r)
        M_dm = self.nfw_profile.mass_enclosed(r)
        
        return self.Omega_ratio * g * M_dm
    
