"""
Compare simulation profile with Churazov et al. 2021 Coma observations.

Usage:
    python compare_churazov.py                  # Single MCMC ID 0
    python compare_churazov.py 0 1 2 3          # Specific MCMC IDs
    python compare_churazov.py 0-5              # Range of MCMC IDs (0 to 5 inclusive)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

from surface_brightness import compute_xray_profile


def load_churazov_data(filename="data/churazov2021_coma_profile.csv"):
    """Load the extracted Churazov+2021 data."""
    data = np.loadtxt(filename, delimiter=",", comments="#")
    return data[:, 0], data[:, 1]


def parse_mcmc_ids(args):
    """Parse MCMC IDs from command line arguments."""
    if not args:
        return [0]  # Default to MCMC ID 0

    mcmc_ids = []
    for arg in args:
        if "-" in arg:
            # Range like "0-5"
            start, end = map(int, arg.split("-"))
            mcmc_ids.extend(range(start, end + 1))
        else:
            mcmc_ids.append(int(arg))

    return sorted(set(mcmc_ids))  # Remove duplicates and sort


def get_snapshot_path(mcmc_id):
    """Get snapshot path for a given MCMC ID."""
    return f"data/mcmc_{mcmc_id}/swift/snap_0077/snap_0077.hdf5"


def compute_and_plot_single(
    mcmc_id,
    r_obs,
    sb_obs,
    cluster_center,
    observer_position,
    max_radius,
    cylinder_depth,
    n_bins,
    effective_area,
    redshift,
    norm_radius_target=5.0,
):
    """
    Compute profile for a single MCMC ID and create 2-panel plot.

    Returns:
        r_sim, sb_sim_normalized, norm_factor for use in summary plot
    """
    snapshot_path = get_snapshot_path(mcmc_id)

    print(f"\n{'='*60}")
    print(f"Processing MCMC ID {mcmc_id}")
    print(f"{'='*60}")
    print(f"Snapshot: {snapshot_path}")

    # Compute simulation profile with 2D image
    r_sim, sb_sim, image_data = compute_xray_profile(
        snapshot_path=snapshot_path,
        cluster_center=cluster_center,
        observer_position=observer_position,
        max_radius=max_radius,
        cylinder_depth=cylinder_depth,
        n_bins=n_bins,
        energy_band="erosita_low",
        log_spacing=True,
        effective_area=effective_area,
        output_units="counts/s/arcmin2",
        redshift=redshift,
        return_image=True,
        n_pixels=512,
    )

    # Determine normalization radius
    norm_radius = max(norm_radius_target, r_sim.min() * 1.5)
    print(f"Simulation radial range: {r_sim.min():.1f} - {r_sim.max():.1f} arcmin")

    # Interpolate for normalization
    sim_interp = interp1d(r_sim, sb_sim, kind="linear", bounds_error=False, fill_value=np.nan)
    obs_interp = interp1d(r_obs, sb_obs, kind="linear", bounds_error=False, fill_value=np.nan)

    sb_sim_at_norm = sim_interp(norm_radius)
    sb_obs_at_norm = obs_interp(norm_radius)

    # Validate normalization values
    if np.isnan(sb_sim_at_norm) or sb_sim_at_norm <= 0:
        print(f"WARNING: Invalid simulation value at {norm_radius} arcmin: {sb_sim_at_norm}")
        return None, None, None, None
    if np.isnan(sb_obs_at_norm) or sb_obs_at_norm <= 0:
        print(f"WARNING: Invalid observation value at {norm_radius} arcmin: {sb_obs_at_norm}")
        return None, None, None, None

    # Normalize
    norm_factor = sb_obs_at_norm / sb_sim_at_norm
    sb_sim_normalized = sb_sim * norm_factor

    print(f"Normalization at {norm_radius:.1f} arcmin:")
    print(f"  Obs: {sb_obs_at_norm:.3e}, Sim: {sb_sim_at_norm:.3e}")
    print(f"  Normalization factor: {norm_factor:.3f}")

    # ==================== Create 2-panel figure ====================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: 2D image
    ax_img = axes[0]
    image = image_data["image"] * norm_factor
    extent = image_data["extent_arcmin"]
    image_plot = np.clip(image, 1e-8, None)

    im = ax_img.imshow(
        image_plot,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=LogNorm(),
        aspect="equal",
    )

    cbar = plt.colorbar(im, ax=ax_img, shrink=0.8, pad=0.02)
    cbar.set_label(r"$I_X$ [counts/s/arcmin$^2$]", fontsize=11)

    # Add R500 and R200 circles
    r500_arcmin, r200_arcmin = 47, 77
    ax_img.add_patch(plt.Circle((0, 0), r500_arcmin, fill=False, color="white", ls="--", lw=1.5))
    ax_img.add_patch(plt.Circle((0, 0), r200_arcmin, fill=False, color="white", ls=":", lw=1.5))

    ax_img.set_xlabel("x [arcmin]", fontsize=12)
    ax_img.set_ylabel("y [arcmin]", fontsize=12)
    ax_img.set_title(f"X-ray Surface Brightness Map (MCMC {mcmc_id})", fontsize=12)
    ax_img.plot([], [], "w--", lw=1.5, label=r"$R_{500}$ (47')")
    ax_img.plot([], [], "w:", lw=1.5, label=r"$R_{200}$ (77')")
    ax_img.legend(loc="upper right", fontsize=9)

    # Right panel: Radial profile
    ax_prof = axes[1]
    ax_prof.loglog(r_obs, sb_obs, "bo", markersize=6, label="Churazov+2021 (eROSITA)")
    ax_prof.loglog(r_sim, sb_sim_normalized, "r-", linewidth=2, label=f"MCMC {mcmc_id} (normalized)")
    ax_prof.axvline(norm_radius, color="green", ls="-", alpha=0.5, lw=1, label=f"Norm @ {norm_radius:.0f}'")
    ax_prof.axvline(47, color="gray", ls="--", alpha=0.5, label=r"$R_{500}$ (47')")
    ax_prof.axvline(77, color="gray", ls=":", alpha=0.5, label=r"$R_{200}$ (77')")

    ax_prof.set_xlabel("r [arcmin]", fontsize=12)
    ax_prof.set_ylabel(r"$I_X$ [counts/s/arcmin$^2$]", fontsize=12)
    ax_prof.set_title(f"Radial Profile Comparison (MCMC {mcmc_id})", fontsize=12)
    ax_prof.legend(loc="upper right", fontsize=9)
    ax_prof.set_xlim(0.3, 300)
    ax_prof.set_ylim(1e-6, 0.2)
    ax_prof.grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = f"comparison_churazov2021_mcmc{mcmc_id}.png"
    plt.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")
    plt.close(fig)

    return r_sim, sb_sim_normalized, norm_radius, norm_factor


def plot_all_profiles(
    r_obs,
    sb_obs,
    all_profiles,
    norm_radius,
):
    """Create summary plot with all MCMC profiles overlaid."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Observational data
    ax.loglog(r_obs, sb_obs, "ko", markersize=8, label="Churazov+2021 (eROSITA)", zorder=10)

    # Plot all MCMC profiles
    cmap = plt.cm.viridis
    n_profiles = len(all_profiles)

    for i, (mcmc_id, r_sim, sb_sim_normalized) in enumerate(all_profiles):
        color = cmap(i / max(1, n_profiles - 1)) if n_profiles > 1 else "red"
        ax.loglog(r_sim, sb_sim_normalized, "-", linewidth=1.5, color=color,
                  label=f"MCMC {mcmc_id}", alpha=0.8)

    # Reference lines
    ax.axvline(norm_radius, color="green", ls="-", alpha=0.5, lw=1, label=f"Norm @ {norm_radius:.0f}'")
    ax.axvline(47, color="gray", ls="--", alpha=0.5, label=r"$R_{500}$ (47')")
    ax.axvline(77, color="gray", ls=":", alpha=0.5, label=r"$R_{200}$ (77')")

    ax.set_xlabel("r [arcmin]", fontsize=12)
    ax.set_ylabel(r"$I_X$ [counts/s/arcmin$^2$]", fontsize=12)
    ax.set_title("X-ray Surface Brightness Profiles - All MCMC Samples", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, ncol=2 if n_profiles > 5 else 1)
    ax.set_xlim(0.3, 300)
    ax.set_ylim(1e-6, 0.2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = "comparison_churazov2021_all_mcmc.png"
    plt.savefig(outfile, dpi=150)
    print(f"\nSaved {outfile}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare simulation profiles with Churazov+2021 Coma observations."
    )
    parser.add_argument(
        "mcmc_ids",
        nargs="*",
        default=[],
        help="MCMC IDs to process. Can be individual IDs (0 1 2) or ranges (0-5)."
    )
    args = parser.parse_args()

    mcmc_ids = parse_mcmc_ids(args.mcmc_ids)
    print(f"Processing MCMC IDs: {mcmc_ids}")

    # Load observational data
    r_obs, sb_obs = load_churazov_data()

    # Simulation parameters (same for all MCMC samples)
    cluster_center = np.array([407.754039, 476.290027, 550.733207])  # Mpc
    observer_position = np.array([500.0, 500.0, 500.0])  # Mpc
    max_radius = 5.0  # Mpc
    cylinder_depth = 10.0  # Mpc
    n_bins = 40
    effective_area = 400.0  # eROSITA single module
    redshift = 0.023
    norm_radius_target = 5.0  # arcmin

    # Process each MCMC ID
    all_profiles = []
    final_norm_radius = norm_radius_target

    for mcmc_id in mcmc_ids:
        try:
            r_sim, sb_sim_normalized, norm_radius, norm_factor = compute_and_plot_single(
                mcmc_id=mcmc_id,
                r_obs=r_obs,
                sb_obs=sb_obs,
                cluster_center=cluster_center,
                observer_position=observer_position,
                max_radius=max_radius,
                cylinder_depth=cylinder_depth,
                n_bins=n_bins,
                effective_area=effective_area,
                redshift=redshift,
                norm_radius_target=norm_radius_target,
            )

            if r_sim is not None:
                all_profiles.append((mcmc_id, r_sim, sb_sim_normalized))
                final_norm_radius = norm_radius

        except Exception as e:
            print(f"ERROR processing MCMC {mcmc_id}: {e}")
            continue

    # Create summary plot with all profiles
    if len(all_profiles) > 0:
        plot_all_profiles(r_obs, sb_obs, all_profiles, final_norm_radius)
    else:
        print("No valid profiles to plot!")

    print(f"\nProcessed {len(all_profiles)}/{len(mcmc_ids)} MCMC samples successfully.")


if __name__ == "__main__":
    main()
