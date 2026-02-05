"""Unit conversion utilities for X-ray surface brightness calculations."""

import numpy as np


# Physical constants
MPC_TO_CM = 3.08567758e24  # cm per Mpc
RAD_TO_ARCMIN = (180.0 / np.pi) * 60.0  # arcmin per radian


def cosmological_dimming(redshift: float) -> float:
    """
    Compute cosmological surface brightness dimming factor.

    Surface brightness dims as (1+z)^4 due to:
    - Photon energy redshift: (1+z)
    - Time dilation (arrival rate): (1+z)
    - Angular diameter distance squared: (1+z)^2

    Parameters:
        redshift: Source redshift

    Returns:
        Dimming factor to multiply intrinsic surface brightness by
    """
    return 1.0 / (1.0 + redshift) ** 4


def mpc_to_arcmin(r_mpc: np.ndarray, distance_mpc: float) -> np.ndarray:
    """
    Convert physical distance in Mpc to angular distance in arcmin.

    Parameters:
        r_mpc: Physical distance(s) in Mpc
        distance_mpc: Distance from observer to object in Mpc

    Returns:
        Angular distance in arcmin
    """
    return (r_mpc / distance_mpc) * RAD_TO_ARCMIN


def arcmin_to_mpc(theta_arcmin: np.ndarray, distance_mpc: float) -> np.ndarray:
    """
    Convert angular distance in arcmin to physical distance in Mpc.

    Parameters:
        theta_arcmin: Angular distance(s) in arcmin
        distance_mpc: Distance from observer to object in Mpc

    Returns:
        Physical distance in Mpc
    """
    return theta_arcmin * distance_mpc / RAD_TO_ARCMIN


def annulus_area_arcmin2(
    theta_inner: np.ndarray, theta_outer: np.ndarray
) -> np.ndarray:
    """
    Compute area of annulus in arcmin^2.

    Parameters:
        theta_inner: Inner radius of annulus in arcmin
        theta_outer: Outer radius of annulus in arcmin

    Returns:
        Area in arcmin^2
    """
    return np.pi * (theta_outer**2 - theta_inner**2)


def photon_luminosity_to_flux(
    photon_luminosity: np.ndarray, distance_mpc: float
) -> np.ndarray:
    """
    Convert photon luminosity to photon flux.

    Parameters:
        photon_luminosity: Photon luminosity in photons/s
        distance_mpc: Distance from observer in Mpc

    Returns:
        Photon flux in photons/s/cm^2
    """
    distance_cm = distance_mpc * MPC_TO_CM
    return photon_luminosity / (4.0 * np.pi * distance_cm**2)
