"""Data loading utilities using swiftsimio."""

from typing import Any

import numpy as np
import swiftsimio as sw
from unyt import unyt_array


# Mapping from user-friendly band names to swiftsimio attribute names
BAND_MAPPING = {
    "erosita_low": "erosita_low",
    "erosita_high": "erosita_high",
    "rosat": "ROSAT",
}


def load_cluster_region(
    snapshot_path: str,
    center: np.ndarray,
    load_radius: float,
) -> Any:
    """
    Load gas particles within a cubic region around a cluster center.

    Uses swiftsimio's spatial masking to efficiently load only particles
    in the region of interest.

    Parameters:
        snapshot_path: Path to SWIFT HDF5 snapshot file
        center: 3D position of cluster center [x, y, z] in Mpc
        load_radius: Half-width of cubic loading region in Mpc

    Returns:
        SWIFTDataset object with loaded gas particles
    """
    mask = sw.mask(snapshot_path)

    # Get box size to ensure we don't request beyond boundaries
    boxsize = mask.metadata.boxsize

    # Define cubic load region around center
    load_region = []
    for i in range(3):
        lower = max(0.0, center[i] - load_radius) * boxsize[0].units
        upper = min(float(boxsize[i].value), center[i] + load_radius) * boxsize[0].units
        load_region.append([lower, upper])

    mask.constrain_spatial(load_region)

    return sw.load(snapshot_path, mask=mask)


def extract_particle_data(
    data: Any,
    energy_band: str = "erosita_low",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract coordinates, smoothing lengths, and X-ray photon luminosities.

    Parameters:
        data: SWIFTDataset object from load_cluster_region
        energy_band: Which X-ray band to use ("erosita_low", "erosita_high", "rosat")

    Returns:
        coords: Particle coordinates in Mpc, shape (N, 3)
        smoothing_lengths: SPH smoothing lengths in Mpc, shape (N,)
        photon_luminosity: X-ray photon luminosity in photons/s, shape (N,)
    """
    if energy_band not in BAND_MAPPING:
        raise ValueError(
            f"Unknown energy band '{energy_band}'. "
            f"Available bands: {list(BAND_MAPPING.keys())}"
        )

    # Get coordinates in Mpc
    coords = data.gas.coordinates.to("Mpc").value

    # Get smoothing lengths in Mpc
    smoothing_lengths = data.gas.smoothing_lengths.to("Mpc").value

    # Get X-ray photon luminosity for the specified band
    xray_photon_lum = data.gas.xray_photon_luminosities
    band_name = BAND_MAPPING[energy_band]
    photon_luminosity_raw = getattr(xray_photon_lum, band_name)

    # Print units for debugging
    print(f"  Raw luminosity units: {photon_luminosity_raw.units}")

    # Check if this is a specific quantity (per unit mass)
    # SWIFT often stores X-ray luminosity as photons/s/mass
    unit_str = str(photon_luminosity_raw.units)
    if "Msun" in unit_str or "g" in unit_str or "mass" in unit_str.lower():
        # Multiply by particle mass to get total luminosity
        print("  Detected specific luminosity - multiplying by particle mass")
        masses = data.gas.masses
        photon_luminosity = photon_luminosity_raw * masses
        photon_luminosity = photon_luminosity.to("1/s").value
    else:
        # Already total luminosity per particle
        photon_luminosity = photon_luminosity_raw.to("1/s").value

    # Print diagnostic info
    print(f"  Total photon luminosity: {np.sum(photon_luminosity):.3e} photons/s")
    print(f"  Max per particle: {np.max(photon_luminosity):.3e} photons/s")

    return coords, smoothing_lengths, photon_luminosity


def get_observer_distance(
    cluster_center: np.ndarray, observer_position: np.ndarray
) -> float:
    """
    Compute distance from observer to cluster center.

    Parameters:
        cluster_center: 3D position of cluster center in Mpc
        observer_position: 3D position of observer in Mpc

    Returns:
        Distance in Mpc
    """
    return np.linalg.norm(np.asarray(cluster_center) - np.asarray(observer_position))
