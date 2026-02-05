"""Radial profile computation for X-ray surface brightness."""

import numpy as np

from .units import (
    annulus_area_arcmin2,
    mpc_to_arcmin,
    photon_luminosity_to_flux,
    cosmological_dimming,
    MPC_TO_CM,
)
from .loader import (
    load_cluster_region,
    extract_particle_data,
    get_observer_distance,
)
from .projection import compute_los_vector, project_particles, compute_2d_image


def compute_radial_bins(
    max_radius_mpc: float,
    n_bins: int,
    distance_mpc: float,
    log_spacing: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute radial bin edges and centers in both Mpc and arcmin.

    Parameters:
        max_radius_mpc: Maximum radius in Mpc
        n_bins: Number of radial bins
        distance_mpc: Distance from observer to cluster in Mpc
        log_spacing: If True, use logarithmic spacing; if False, linear

    Returns:
        bin_edges_mpc: Bin edges in Mpc, shape (n_bins + 1,)
        bin_edges_arcmin: Bin edges in arcmin, shape (n_bins + 1,)
        bin_centers_arcmin: Bin centers in arcmin, shape (n_bins,)
    """
    if log_spacing:
        # Use logarithmic spacing, starting from a small fraction of max
        min_radius = max_radius_mpc / 100.0
        bin_edges_mpc = np.logspace(
            np.log10(min_radius), np.log10(max_radius_mpc), n_bins + 1
        )
    else:
        bin_edges_mpc = np.linspace(0.0, max_radius_mpc, n_bins + 1)

    bin_edges_arcmin = mpc_to_arcmin(bin_edges_mpc, distance_mpc)

    # Bin centers (geometric mean for log spacing, arithmetic for linear)
    if log_spacing:
        bin_centers_arcmin = np.sqrt(bin_edges_arcmin[:-1] * bin_edges_arcmin[1:])
    else:
        bin_centers_arcmin = 0.5 * (bin_edges_arcmin[:-1] + bin_edges_arcmin[1:])

    return bin_edges_mpc, bin_edges_arcmin, bin_centers_arcmin


def bin_photon_luminosity(
    r_perp: np.ndarray,
    photon_luminosity: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """
    Bin photon luminosity by projected radius.

    Parameters:
        r_perp: Projected distances in Mpc, shape (N,)
        photon_luminosity: Photon luminosities in photons/s, shape (N,)
        bin_edges: Radial bin edges in Mpc, shape (n_bins + 1,)

    Returns:
        binned_luminosity: Total photon luminosity in each bin, photons/s
    """
    bin_indices = np.digitize(r_perp, bin_edges)
    n_bins = len(bin_edges) - 1

    binned_luminosity = np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_indices == (i + 1)
        binned_luminosity[i] = np.sum(photon_luminosity[mask])

    return binned_luminosity


def compute_profile_from_image(
    image: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    bin_edges_mpc: np.ndarray,
) -> np.ndarray:
    """
    Compute azimuthally averaged radial profile from 2D image.

    Parameters:
        image: 2D luminosity density image (photons/s/Mpc^2)
        x_centers: Pixel x centers in Mpc
        y_centers: Pixel y centers in Mpc
        bin_edges_mpc: Radial bin edges in Mpc

    Returns:
        binned_luminosity: Total luminosity in each radial bin (photons/s)
    """
    # Create 2D radius array
    xx, yy = np.meshgrid(x_centers, y_centers)
    r_2d = np.sqrt(xx**2 + yy**2)

    # Pixel area
    dx = x_centers[1] - x_centers[0]
    dy = y_centers[1] - y_centers[0]
    pixel_area = dx * dy

    # Bin by radius
    n_bins = len(bin_edges_mpc) - 1
    binned_luminosity = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (r_2d >= bin_edges_mpc[i]) & (r_2d < bin_edges_mpc[i + 1])
        # image is luminosity density, multiply by pixel area to get luminosity
        binned_luminosity[i] = np.sum(image[mask]) * pixel_area

    return binned_luminosity


def luminosity_to_surface_brightness(
    binned_luminosity: np.ndarray,
    bin_edges_arcmin: np.ndarray,
    distance_mpc: float,
    effective_area: float = 1.0,
    output_units: str = "counts/s/arcmin2",
) -> np.ndarray:
    """
    Convert binned photon luminosity to surface brightness.

    Parameters:
        binned_luminosity: Total photon luminosity per bin in photons/s
        bin_edges_arcmin: Bin edges in arcmin
        distance_mpc: Distance from observer in Mpc
        effective_area: Telescope effective area in cm^2 (default: 1.0)
            - eROSITA (single module): ~400 cm^2 at 1 keV
            - eROSITA (7 modules): ~2400 cm^2 at 1 keV
            - XMM-Newton EPIC-pn: ~1000 cm^2 at 1 keV
        output_units: Output units, one of:
            - "counts/s/arcmin2": counts per second per arcmin^2
            - "counts/s/sr": counts per second per steradian (Churazov+2021 style)
            - "photons/s/cm2/arcmin2": raw photon flux (no effective area)

    Returns:
        surface_brightness: Surface brightness in specified units
    """
    # Compute flux (photons/s/cm^2) from luminosity
    binned_flux = photon_luminosity_to_flux(binned_luminosity, distance_mpc)

    # Compute annulus areas in arcmin^2
    areas_arcmin2 = annulus_area_arcmin2(bin_edges_arcmin[:-1], bin_edges_arcmin[1:])

    if output_units == "photons/s/cm2/arcmin2":
        # Raw photon flux per arcmin^2
        surface_brightness = binned_flux / areas_arcmin2

    elif output_units == "counts/s/arcmin2":
        # Multiply by effective area to get count rate
        # flux [photons/s/cm^2] * A_eff [cm^2] = count_rate [counts/s]
        binned_count_rate = binned_flux * effective_area
        surface_brightness = binned_count_rate / areas_arcmin2

    elif output_units == "counts/s/sr":
        # Convert to per steradian (Churazov+2021 convention)
        # 1 sr = (180/pi)^2 * 3600 arcmin^2 = 4.2545e7 arcmin^2
        ARCMIN2_PER_SR = (180.0 / np.pi) ** 2 * 3600.0
        binned_count_rate = binned_flux * effective_area
        surface_brightness = binned_count_rate / areas_arcmin2 * ARCMIN2_PER_SR

    else:
        raise ValueError(
            f"Unknown output_units '{output_units}'. "
            "Use 'counts/s/arcmin2', 'counts/s/sr', or 'photons/s/cm2/arcmin2'"
        )

    return surface_brightness


def compute_xray_profile(
    snapshot_path: str,
    cluster_center: np.ndarray,
    observer_position: np.ndarray,
    max_radius: float,
    cylinder_depth: float,
    n_bins: int = 50,
    energy_band: str = "erosita_low",
    log_spacing: bool = True,
    effective_area: float = 1.0,
    output_units: str = "counts/s/arcmin2",
    redshift: float = 0.0,
    return_image: bool = False,
    n_pixels: int = 512,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute X-ray surface brightness profile from a SWIFT snapshot.

    Uses SPH kernel smoothing for accurate projection.

    Parameters:
        snapshot_path: Path to SWIFT HDF5 snapshot file
        cluster_center: 3D position of cluster center [x, y, z] in Mpc
        observer_position: 3D position of observer [x, y, z] in Mpc
        max_radius: Maximum projected radius for profile in Mpc
        cylinder_depth: Depth of integration cylinder along LOS in Mpc
        n_bins: Number of radial bins (default: 50)
        energy_band: X-ray band to use ("erosita_low", "erosita_high", "rosat")
        log_spacing: Use logarithmic bin spacing (default: True)
        effective_area: Telescope effective area in cm^2 (default: 1.0)
            Common values:
            - eROSITA single module: ~400 cm^2 at 1 keV
            - eROSITA 7 modules: ~2400 cm^2 at 1 keV
            - XMM-Newton EPIC-pn: ~1000 cm^2 at 1 keV
        output_units: Output units (default: "counts/s/arcmin2")
            - "counts/s/arcmin2": counts per second per arcmin^2
            - "counts/s/sr": counts per second per steradian
            - "photons/s/cm2/arcmin2": raw photon flux
        redshift: Source redshift for cosmological dimming (default: 0.0)
        return_image: If True, also return the 2D surface brightness image
        n_pixels: Number of pixels per side for 2D image (default: 512)

    Returns:
        radii: Bin centers in arcmin, shape (n_bins,)
        surface_brightness: Surface brightness in specified units, shape (n_bins,)
        image_data: (if return_image=True) Dictionary containing:
            - "image": 2D surface brightness array
            - "extent_arcmin": [xmin, xmax, ymin, ymax] in arcmin
            - "extent_mpc": [xmin, xmax, ymin, ymax] in Mpc
    """
    # Compute observer distance
    distance_mpc = get_observer_distance(cluster_center, observer_position)

    # Load region large enough to encompass cylinder
    # Need to load slightly larger region to account for cylinder depth and smoothing
    load_radius = max(max_radius, cylinder_depth / 2.0) * 1.2

    print(f"Loading particles within {load_radius:.2f} Mpc of cluster center...")
    data = load_cluster_region(snapshot_path, cluster_center, load_radius)

    # Extract particle data
    print(f"Extracting particle data for {energy_band} band...")
    coords, smoothing_lengths, photon_luminosity = extract_particle_data(
        data, energy_band
    )
    print(f"Loaded {len(coords)} particles")

    # Compute LOS vector
    los_vector = compute_los_vector(cluster_center, observer_position)

    # Compute 2D image with SPH kernel smoothing
    print(f"Computing 2D projection with SPH smoothing ({n_pixels}x{n_pixels})...")
    image_lum_density, x_centers, y_centers = compute_2d_image(
        coords,
        cluster_center,
        los_vector,
        smoothing_lengths,
        photon_luminosity,
        image_half_width=max_radius,
        n_pixels=n_pixels,
        cylinder_depth=cylinder_depth,
    )

    # Compute radial bins
    bin_edges_mpc, bin_edges_arcmin, bin_centers_arcmin = compute_radial_bins(
        max_radius, n_bins, distance_mpc, log_spacing
    )

    # Compute radial profile from image
    print("Computing radial profile from image...")
    binned_luminosity = compute_profile_from_image(
        image_lum_density, x_centers, y_centers, bin_edges_mpc
    )

    # Convert to surface brightness
    surface_brightness = luminosity_to_surface_brightness(
        binned_luminosity,
        bin_edges_arcmin,
        distance_mpc,
        effective_area=effective_area,
        output_units=output_units,
    )

    # Apply cosmological dimming
    if redshift > 0:
        dimming = cosmological_dimming(redshift)
        print(f"Applying cosmological dimming for z={redshift}: factor = {dimming:.4f}")
        surface_brightness = surface_brightness * dimming

    if return_image:
        # Convert image to surface brightness units
        # image_lum_density is in photons/s/Mpc^2
        # Convert to photons/s/cm^2/Mpc^2, then to per arcmin^2

        # First convert Mpc^2 to cm^2 in denominator
        # photons/s/Mpc^2 * (1 Mpc / MPC_TO_CM cm)^2 = photons/s/cm^2 * Mpc^2/cm^2
        # Wait, we need flux, not luminosity density

        # The image is luminosity per area (photons/s/Mpc^2)
        # To get flux: divide by 4*pi*D^2
        distance_cm = distance_mpc * MPC_TO_CM
        image_flux_density = image_lum_density / (4.0 * np.pi * distance_cm**2)
        # Now in photons/s/cm^2/Mpc^2

        # Convert Mpc^2 to arcmin^2
        # 1 Mpc at distance D subtends theta = 1/D radians = RAD_TO_ARCMIN/D arcmin
        # So 1 Mpc^2 = (RAD_TO_ARCMIN/D)^2 arcmin^2
        from .units import RAD_TO_ARCMIN

        mpc2_to_arcmin2 = (RAD_TO_ARCMIN / distance_mpc) ** 2
        image_sb = image_flux_density * mpc2_to_arcmin2  # photons/s/cm^2/arcmin^2

        # Apply effective area if needed
        if output_units == "counts/s/arcmin2":
            image_sb = image_sb * effective_area
        elif output_units == "counts/s/sr":
            ARCMIN2_PER_SR = (180.0 / np.pi) ** 2 * 3600.0
            image_sb = image_sb * effective_area * ARCMIN2_PER_SR

        # Apply cosmological dimming to image
        if redshift > 0:
            image_sb = image_sb * dimming

        # Compute extent in arcmin
        extent_mpc = [
            x_centers[0] - 0.5 * (x_centers[1] - x_centers[0]),
            x_centers[-1] + 0.5 * (x_centers[1] - x_centers[0]),
            y_centers[0] - 0.5 * (y_centers[1] - y_centers[0]),
            y_centers[-1] + 0.5 * (y_centers[1] - y_centers[0]),
        ]
        extent_arcmin = [mpc_to_arcmin(e, distance_mpc) for e in extent_mpc]

        image_data = {
            "image": image_sb,
            "extent_arcmin": extent_arcmin,
            "extent_mpc": extent_mpc,
        }

        return bin_centers_arcmin, surface_brightness, image_data

    return bin_centers_arcmin, surface_brightness
