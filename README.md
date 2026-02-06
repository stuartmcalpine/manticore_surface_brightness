# Manticore Surface Brightness

Compute synthetic X-ray surface brightness profiles from [SWIFT](https://swift.dur.ac.uk/) cosmological hydrodynamic simulation snapshots and compare them to observational data.

This package was developed for analysing simulated galaxy clusters (specifically the Coma cluster) against eROSITA observations from Churazov et al. (2021).

## Overview

The pipeline performs the following steps:

1. Load gas particle data from a SWIFT HDF5 snapshot within a region around the cluster centre.
2. Project particles onto a 2D plane perpendicular to the observer line of sight using SPH kernel smoothing.
3. Compute an azimuthally averaged radial surface brightness profile from the projected image.
4. Convert from intrinsic photon luminosity to observable surface brightness units, accounting for telescope effective area and cosmological dimming.

## Installation

### Dependencies

- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- [swiftsimio](https://github.com/SWIFTSIM/swiftsimio)
- [unyt](https://github.com/yt-project/unyt)
- [Numba](https://numba.pydata.org/)

### Setup

```bash
pip install -e .
```

## Usage

### As a library

```python
import numpy as np
from surface_brightness import compute_xray_profile

radii, sb = compute_xray_profile(
    snapshot_path="data/mcmc_0/swift/snap_0077/snap_0077.hdf5",
    cluster_center=np.array([407.75, 476.29, 550.73]),  # Mpc
    observer_position=np.array([500.0, 500.0, 500.0]),   # Mpc
    max_radius=5.0,        # Mpc
    cylinder_depth=10.0,   # Mpc
    n_bins=40,
    energy_band="erosita_low",
    effective_area=400.0,  # cm^2
    output_units="counts/s/arcmin2",
    redshift=0.023,
)
```

The function also supports returning the full 2D surface brightness image:

```python
radii, sb, image_data = compute_xray_profile(
    ...,
    return_image=True,
    n_pixels=512,
)
```

### Comparison with observations

The `compare_churazov.py` script compares simulated profiles against the Churazov et al. (2021) eROSITA Coma cluster observations:

```bash
# Single MCMC sample (default: ID 0)
python compare_churazov.py

# Specific MCMC sample IDs
python compare_churazov.py 0 1 2 3

# Range of MCMC sample IDs
python compare_churazov.py 0-5
```

For each MCMC sample, the script produces a two-panel figure showing the 2D X-ray map and the radial profile comparison. When multiple samples are processed, a summary plot with all profiles overlaid is also generated.

## Method

### Coordinate projection

Given a cluster centre position $\mathbf{c}$ and an observer position $\mathbf{o}$, the line-of-sight (LOS) unit vector is

$$\hat{\mathbf{n}} = \frac{\mathbf{c} - \mathbf{o}}{|\mathbf{c} - \mathbf{o}|}$$

An orthonormal basis $(\hat{\mathbf{x}}, \hat{\mathbf{y}})$ for the projection plane is constructed via Gram--Schmidt orthogonalisation against $\hat{\mathbf{n}}$. Each particle's position relative to the cluster centre, $\Delta\mathbf{r} = \mathbf{r} - \mathbf{c}$, is decomposed into:

- LOS distance: $d_\parallel = \Delta\mathbf{r} \cdot \hat{\mathbf{n}}$
- Projected coordinates: $x = \Delta\mathbf{r} \cdot \hat{\mathbf{x}}$, $\quad y = \Delta\mathbf{r} \cdot \hat{\mathbf{y}}$

Only particles within a cylinder of half-depth $D/2$ along the LOS are retained:

$$|d_\parallel| < D / 2$$

### SPH kernel smoothing

Each particle's X-ray photon luminosity $L_i$ (in photons/s) is distributed onto a 2D pixel grid using a Gaussian kernel that approximates the projected cubic spline SPH kernel:

$$W(r, h) = \exp\!\left(-\frac{r^2}{2\sigma^2}\right), \quad \sigma = h/2$$

where $h$ is the SPH smoothing length and $r$ is the distance from the particle centre to a pixel centre. The kernel is truncated at $3\sigma = 1.5h$.

The luminosity deposited by particle $i$ in pixel $j$ is

$$\ell_{ij} = L_i \, \frac{W(r_{ij},\, h_i)}{\displaystyle\sum_k W(r_{ik},\, h_i)}$$

so that the total luminosity is conserved: $\sum_j \ell_{ij} = L_i$. The resulting image stores luminosity surface density in units of photons/s/Mpc$^2$.

The kernel scattering loop is JIT-compiled and parallelised with Numba for performance.

### Radial profile extraction

The 2D image is azimuthally averaged into radial bins. Bin edges can be spaced linearly or logarithmically. For logarithmic spacing, the minimum radius is set to $R_\mathrm{max}/100$ and bin centres are computed as the geometric mean of adjacent edges:

$$r_i = \sqrt{r_{\mathrm{inner},i} \, r_{\mathrm{outer},i}}$$

The total luminosity in each radial bin is obtained by summing pixel values within the annulus and multiplying by the pixel area.

### Unit conversions

#### Angular--physical conversion

Physical distances are converted to angular distances via the small-angle approximation:

$$\theta\;\text{[arcmin]} = \frac{r\;\text{[Mpc]}}{D\;\text{[Mpc]}} \times \frac{180 \times 60}{\pi}$$

where $D$ is the observer--cluster distance.

#### Photon flux

The photon luminosity in each annular bin $L_\mathrm{bin}$ (photons/s) is converted to photon flux at the observer:

$$F = \frac{L_\mathrm{bin}}{4\pi D_\mathrm{cm}^2} \quad \text{[photons/s/cm}^2\text{]}$$

where $D_\mathrm{cm} = D \times 3.086 \times 10^{24}$ cm/Mpc.

#### Surface brightness

The photon flux is divided by the annulus area to obtain a surface brightness. Three output unit conventions are supported:

| Unit | Formula |
|------|---------|
| `photons/s/cm2/arcmin2` | $I = F / A_\mathrm{arcmin^2}$ |
| `counts/s/arcmin2` | $I = F \cdot A_\mathrm{eff} / A_\mathrm{arcmin^2}$ |
| `counts/s/sr` | $I = F \cdot A_\mathrm{eff} / A_\mathrm{arcmin^2} \times \Omega_\mathrm{sr}$ |

where $A_\mathrm{eff}$ is the telescope effective area in cm$^2$, $A_\mathrm{arcmin^2} = \pi(r_\mathrm{out}^2 - r_\mathrm{in}^2)$ is the annulus area, and $\Omega_\mathrm{sr} = (180/\pi)^2 \times 3600$ arcmin$^2$/sr.

#### Cosmological dimming

For sources at redshift $z > 0$, the surface brightness is reduced by a factor of

$$\mathcal{D}(z) = (1 + z)^{-4}$$

which accounts for photon energy redshift, time dilation of the photon arrival rate, and the angular diameter distance scaling.

### Profile normalisation (comparison script)

When comparing to the Churazov et al. (2021) observations, the simulated profile is normalised to match the observed surface brightness at a reference radius $r_\mathrm{norm}$ (default 5 arcmin). Both profiles are interpolated at $r_\mathrm{norm}$ and the normalisation factor is

$$f = \frac{I_\mathrm{obs}(r_\mathrm{norm})}{I_\mathrm{sim}(r_\mathrm{norm})}$$

The normalised simulation profile is then $I_\mathrm{sim}^\prime(r) = f \cdot I_\mathrm{sim}(r)$.

## Package structure

```
surface_brightness/
    __init__.py        # Public API exports
    loader.py          # SWIFT snapshot loading and particle data extraction
    projection.py      # LOS projection and SPH kernel image construction
    profile.py         # Radial profile computation and unit conversion
    units.py           # Physical and angular unit conversion utilities
compare_churazov.py    # Observational comparison script
data/                  # Simulation snapshots and observational data
```

## Supported X-ray bands

| Band name      | Description                          |
|----------------|--------------------------------------|
| `erosita_low`  | eROSITA soft band                    |
| `erosita_high` | eROSITA hard band                    |
| `rosat`        | ROSAT PSPC band                      |

## References

- Churazov, E. et al. (2021). SRG/eROSITA observations of the Coma cluster.
- Schaller, M. et al. (2024). SWIFT: A modern highly-parallel gravity and smoothed particle hydrodynamics solver for astrophysical and cosmological applications.
