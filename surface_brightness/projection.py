"""Line-of-sight projection utilities."""

import numpy as np
from numba import njit, prange


@njit
def _gaussian_kernel_2d(r2: float, h: float) -> float:
    """
    2D Gaussian kernel approximation for projected SPH smoothing.

    Uses sigma = h/2 which approximates the projected cubic spline kernel.

    Parameters:
        r2: Squared distance from particle center
        h: SPH smoothing length

    Returns:
        Kernel weight (unnormalized)
    """
    sigma2 = (h / 2.0) ** 2
    # Truncate at 3 sigma (i.e., 1.5h)
    if r2 > 9.0 * sigma2:
        return 0.0
    return np.exp(-0.5 * r2 / sigma2)


@njit(parallel=True)
def _scatter_to_image(
    image: np.ndarray,
    x_proj: np.ndarray,
    y_proj: np.ndarray,
    smoothing_lengths: np.ndarray,
    luminosity: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    """
    Scatter particle luminosities onto 2D image using SPH kernel smoothing.

    Parameters:
        image: Output image array (modified in place)
        x_proj: Projected x coordinates in Mpc
        y_proj: Projected y coordinates in Mpc
        smoothing_lengths: SPH smoothing lengths in Mpc
        luminosity: Particle luminosities
        x_edges: Pixel x edges
        y_edges: Pixel y edges
    """
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    pixel_area = dx * dy

    n_particles = len(x_proj)

    for p in prange(n_particles):
        xp = x_proj[p]
        yp = y_proj[p]
        h = smoothing_lengths[p]
        lum = luminosity[p]

        # Kernel truncation radius (3 sigma = 1.5h)
        kernel_radius = 1.5 * h

        # Find pixel range affected by this particle
        i_min = int((xp - kernel_radius - x_edges[0]) / dx)
        i_max = int((xp + kernel_radius - x_edges[0]) / dx) + 1
        j_min = int((yp - kernel_radius - y_edges[0]) / dy)
        j_max = int((yp + kernel_radius - y_edges[0]) / dy) + 1

        # Clamp to image bounds
        i_min = max(0, i_min)
        i_max = min(nx, i_max)
        j_min = max(0, j_min)
        j_max = min(ny, j_max)

        # Compute kernel weights for normalization
        total_weight = 0.0
        for i in range(i_min, i_max):
            x_pix = 0.5 * (x_edges[i] + x_edges[i + 1])
            for j in range(j_min, j_max):
                y_pix = 0.5 * (y_edges[j] + y_edges[j + 1])
                r2 = (x_pix - xp) ** 2 + (y_pix - yp) ** 2
                total_weight += _gaussian_kernel_2d(r2, h)

        # Distribute luminosity to pixels
        if total_weight > 0:
            for i in range(i_min, i_max):
                x_pix = 0.5 * (x_edges[i] + x_edges[i + 1])
                for j in range(j_min, j_max):
                    y_pix = 0.5 * (y_edges[j] + y_edges[j + 1])
                    r2 = (x_pix - xp) ** 2 + (y_pix - yp) ** 2
                    w = _gaussian_kernel_2d(r2, h)
                    # Add luminosity density (luminosity per area)
                    image[j, i] += (w / total_weight) * lum / pixel_area


def compute_2d_image(
    coords: np.ndarray,
    cluster_center: np.ndarray,
    los_vector: np.ndarray,
    smoothing_lengths: np.ndarray,
    luminosity: np.ndarray,
    image_half_width: float,
    n_pixels: int,
    cylinder_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D projected image with SPH kernel smoothing.

    Parameters:
        coords: Particle coordinates, shape (N, 3) in Mpc
        cluster_center: 3D position of cluster center in Mpc
        los_vector: Unit vector along line of sight
        smoothing_lengths: SPH smoothing lengths in Mpc
        luminosity: Particle luminosities in photons/s
        image_half_width: Half-width of image in Mpc
        n_pixels: Number of pixels per side
        cylinder_depth: Depth along LOS to include in Mpc

    Returns:
        image: 2D luminosity density image (photons/s/Mpc^2)
        x_centers: Pixel x centers in Mpc (relative to cluster center)
        y_centers: Pixel y centers in Mpc (relative to cluster center)
    """
    # Shift coordinates to cluster-centered frame
    pos_shifted = coords - np.asarray(cluster_center)

    # Compute distance along LOS
    d_los = np.dot(pos_shifted, los_vector)

    # Filter particles within cylinder depth
    within_depth = np.abs(d_los) < (cylinder_depth / 2.0)

    # Build orthonormal basis for projection plane
    # Choose arbitrary perpendicular vector
    if abs(los_vector[2]) < 0.9:
        perp = np.array([0.0, 0.0, 1.0])
    else:
        perp = np.array([1.0, 0.0, 0.0])

    # Gram-Schmidt to get x_hat perpendicular to LOS
    x_hat = perp - np.dot(perp, los_vector) * los_vector
    x_hat = x_hat / np.linalg.norm(x_hat)

    # y_hat = los x x_hat
    y_hat = np.cross(los_vector, x_hat)

    # Project positions onto plane
    x_proj = np.dot(pos_shifted, x_hat)
    y_proj = np.dot(pos_shifted, y_hat)

    # Filter by cylinder depth and image bounds
    within_image = (
        within_depth
        & (np.abs(x_proj) < image_half_width * 1.5)
        & (np.abs(y_proj) < image_half_width * 1.5)
    )

    x_proj_filt = x_proj[within_image]
    y_proj_filt = y_proj[within_image]
    h_filt = smoothing_lengths[within_image]
    lum_filt = luminosity[within_image]

    # Create image grid
    x_edges = np.linspace(-image_half_width, image_half_width, n_pixels + 1)
    y_edges = np.linspace(-image_half_width, image_half_width, n_pixels + 1)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Initialize image
    image = np.zeros((n_pixels, n_pixels), dtype=np.float64)

    # Scatter particles to image
    _scatter_to_image(
        image, x_proj_filt, y_proj_filt, h_filt, lum_filt, x_edges, y_edges
    )

    return image, x_centers, y_centers


def compute_los_vector(
    cluster_center: np.ndarray, observer_position: np.ndarray
) -> np.ndarray:
    """
    Compute unit vector along line of sight from observer to cluster.

    Parameters:
        cluster_center: 3D position of cluster center in Mpc
        observer_position: 3D position of observer in Mpc

    Returns:
        Unit vector pointing from observer to cluster
    """
    los = np.asarray(cluster_center) - np.asarray(observer_position)
    return los / np.linalg.norm(los)


def project_particles(
    coords: np.ndarray,
    cluster_center: np.ndarray,
    los_vector: np.ndarray,
    cylinder_depth: float,
    max_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project particles onto plane perpendicular to line of sight.

    Filters particles to those within a cylinder of specified depth and radius.

    Parameters:
        coords: Particle coordinates, shape (N, 3) in Mpc
        cluster_center: 3D position of cluster center in Mpc
        los_vector: Unit vector along line of sight
        cylinder_depth: Depth of cylinder along LOS in Mpc
        max_radius: Maximum projected radius in Mpc

    Returns:
        r_perp: Projected (impact parameter) distances in Mpc
        mask: Boolean mask of particles within the cylinder
    """
    # Shift coordinates so cluster center is at origin
    pos_shifted = coords - np.asarray(cluster_center)

    # Compute distance along LOS (dot product with LOS vector)
    d_los = np.dot(pos_shifted, los_vector)

    # Compute perpendicular distance (impact parameter)
    # r_perp = |pos - (pos . los) * los|
    pos_parallel = np.outer(d_los, los_vector)
    pos_perp = pos_shifted - pos_parallel
    r_perp = np.linalg.norm(pos_perp, axis=1)

    # Create mask for particles within cylinder
    within_depth = np.abs(d_los) < (cylinder_depth / 2.0)
    within_radius = r_perp < max_radius

    mask = within_depth & within_radius

    return r_perp, mask
