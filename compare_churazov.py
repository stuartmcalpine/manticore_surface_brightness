"""
Compare simulation profile with Churazov et al. 2021 Coma observations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from surface_brightness import compute_xray_profile


def load_churazov_data(filename="data/churazov2021_coma_profile.csv"):
    """Load the extracted Churazov+2021 data."""
    data = np.loadtxt(filename, delimiter=",", comments="#")
    return data[:, 0], data[:, 1]


def main():
    # Load Churazov+2021 observational data
    r_obs, sb_obs = load_churazov_data()

    # Simulation parameters
    snapshot_path = "data/mcmc_0/swift/snap_0077/snap_0077.hdf5"
    cluster_center = np.array([407.754039, 476.290027, 550.733207])  # Mpc
    observer_position = np.array([500.0, 500.0, 500.0])  # Mpc

    # Profile parameters
    max_radius = 5.0  # Mpc
    cylinder_depth = 10.0  # Mpc
    n_bins = 40

    # eROSITA single module effective area (~400 cm^2 at 1 keV)
    # Churazov+2021 normalized per single telescope module
    effective_area = 400.0

    # Coma cluster redshift
    redshift = 0.023

    # Compute simulation profile with 2D image
    print("Computing simulation profile...")
    r_sim, sb_sim, image_data = compute_xray_profile(
        snapshot_path=snapshot_path,
        cluster_center=cluster_center,
        observer_position=observer_position,
        max_radius=max_radius,
        cylinder_depth=cylinder_depth,
        n_bins=n_bins,
        energy_band="erosita_low",  # Closest to 0.4-2 keV
        log_spacing=True,
        effective_area=effective_area,
        output_units="counts/s/arcmin2",
        redshift=redshift,
        return_image=True,
        n_pixels=512,
    )

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ==================== Left panel: 2D image ====================
    ax_img = axes[0]

    # Get image data
    image = image_data["image"]
    extent = image_data["extent_arcmin"]

    # Plot with log color scale
    # Set minimum to avoid log(0)
    image_plot = np.clip(image, 1e-8, None)

    im = ax_img.imshow(
        image_plot,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=LogNorm(),
        aspect="equal",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_img, shrink=0.8, pad=0.02)
    cbar.set_label(r"$I_X$ [counts/s/arcmin$^2$]", fontsize=11)

    # Add R500 and R200 circles
    r500_arcmin = 47
    r200_arcmin = 77
    circle_r500 = plt.Circle(
        (0, 0), r500_arcmin, fill=False, color="white", ls="--", lw=1.5, label=r"$R_{500}$"
    )
    circle_r200 = plt.Circle(
        (0, 0), r200_arcmin, fill=False, color="white", ls=":", lw=1.5, label=r"$R_{200}$"
    )
    ax_img.add_patch(circle_r500)
    ax_img.add_patch(circle_r200)

    ax_img.set_xlabel("x [arcmin]", fontsize=12)
    ax_img.set_ylabel("y [arcmin]", fontsize=12)
    ax_img.set_title("X-ray Surface Brightness Map", fontsize=12)

    # Add legend for circles
    ax_img.plot([], [], "w--", lw=1.5, label=r"$R_{500}$ (47')")
    ax_img.plot([], [], "w:", lw=1.5, label=r"$R_{200}$ (77')")
    ax_img.legend(loc="upper right", fontsize=9)

    # ==================== Right panel: Radial profile ====================
    ax_prof = axes[1]

    # Observational data
    ax_prof.loglog(r_obs, sb_obs, "bo", markersize=6, label="Churazov+2021 (eROSITA)")

    # Simulation
    ax_prof.loglog(r_sim, sb_sim, "r-", linewidth=2, label="Simulation (erosita_low)")

    # Reference lines for Coma
    ax_prof.axvline(47, color="gray", ls="--", alpha=0.5, label=r"$R_{500}$ (47')")
    ax_prof.axvline(77, color="gray", ls=":", alpha=0.5, label=r"$R_{200}$ (77')")

    # Background level from Churazov+2021
    ax_prof.axhline(4e-4, color="black", ls="--", alpha=0.3, label="Background")

    ax_prof.set_xlabel("r [arcmin]", fontsize=12)
    ax_prof.set_ylabel(r"$I_X$ [counts/s/arcmin$^2$]", fontsize=12)
    ax_prof.set_title("Radial Profile Comparison", fontsize=12)
    ax_prof.legend(loc="upper right", fontsize=9)
    ax_prof.set_xlim(0.3, 300)
    ax_prof.set_ylim(1e-6, 0.2)
    ax_prof.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparison_churazov2021.png", dpi=150)
    print("Saved comparison_churazov2021.png")
    plt.show()

    # Print some statistics
    print("\n" + "=" * 60)
    print("COMPARISON STATISTICS")
    print("=" * 60)
    print(f"Redshift: {redshift}")
    print(f"Cosmological dimming factor: {1/(1+redshift)**4:.4f}")

    # Find overlapping radii
    r_min = max(r_sim.min(), r_obs.min())
    r_max = min(r_sim.max(), r_obs.max())

    print(f"Overlapping radius range: {r_min:.1f} - {r_max:.1f} arcmin")

    # Interpolate simulation to observation radii for comparison
    from scipy.interpolate import interp1d

    sim_interp = interp1d(r_sim, sb_sim, kind="linear", fill_value="extrapolate")

    obs_mask = (r_obs >= r_min) & (r_obs <= r_max)
    r_compare = r_obs[obs_mask]
    sb_obs_compare = sb_obs[obs_mask]
    sb_sim_compare = sim_interp(r_compare)

    ratio = sb_sim_compare / sb_obs_compare
    print(f"Ratio (sim/obs) at overlapping radii:")
    print(f"  Mean: {np.mean(ratio):.2f}")
    print(f"  Median: {np.median(ratio):.2f}")
    print(f"  Min: {np.min(ratio):.2f}")
    print(f"  Max: {np.max(ratio):.2f}")


if __name__ == "__main__":
    main()
