"""
X-ray Surface Brightness Profile Calculator

Compute radial X-ray surface brightness profiles from SWIFT simulation snapshots.
"""

from .profile import compute_xray_profile
from .loader import load_cluster_region, extract_particle_data, BAND_MAPPING
from .projection import compute_los_vector, project_particles
from .units import mpc_to_arcmin, arcmin_to_mpc, annulus_area_arcmin2, cosmological_dimming
from .projection import compute_2d_image

__all__ = [
    "compute_xray_profile",
    "load_cluster_region",
    "extract_particle_data",
    "compute_los_vector",
    "project_particles",
    "compute_2d_image",
    "mpc_to_arcmin",
    "arcmin_to_mpc",
    "annulus_area_arcmin2",
    "cosmological_dimming",
    "BAND_MAPPING",
]
