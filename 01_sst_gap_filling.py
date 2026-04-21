# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reproducing Cross Scattering Transform for SST Gap-Filling
#
# ## What this notebook does
#
# This notebook reproduces the application of **Cross Scattering Transform**
# to fill cloud gaps in Sea Surface Temperature (SST) satellite imagery
# from the Copernicus Marine Service.
#
# The scattering transform was originally developed for astrophysics — to
# denoise Planck dust polarisation maps
# ([Delouis et al. 2022, A&A](https://doi.org/10.1051/0004-6361/202244566)).
# Jean-Marc Delouis then demonstrated that the same mathematical framework
# transfers to Earth Observation: filling cloud-obscured regions in
# Copernicus Marine SST data
# ([Pangeo IGARSS 2024 demo](https://pangeo-data.github.io/egi2024-demo/SST_AI.html)).
#
# ## The cross-domain story
#
# | | Astrophysics | Environmental Sciences |
# |---|---|---|
# | **Data** | Planck 353 GHz dust maps | Copernicus Marine SST |
# | **Problem** | Instrument noise | Cloud gaps |
# | **Grid** | HEALPix (sphere) | HEALPix (sphere) |
# | **Method** | Cross Scattering Transform | Cross Scattering Transform |
# | **Tool** | FOSCAT | FOSCAT |
#
# The same method, same grid system, same software — applied to
# fundamentally different scientific domains.
#
# ## Credits
#
# - **Method**: Jean-Marc Delouis, LOPS/CNRS
# - **Original paper**: Delouis et al. (2022), A&A 668, A122
# - **EO application notebook**: Jean-Marc Delouis (Pangeo IGARSS 2024)
# - **This reproduction**: Anne Fouilloux, LifeWatch ERIC (FIESTA-OSCARS)

# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings('ignore')

import xarray as xr
import numpy as np
import healpy as hp
import foscat.scat_cov as sc
import foscat.Synthesis as synthe
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# %% [markdown]
# ## Configuration

# %%
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

TIME_SLICE = slice('2024-06-01', '2024-06-01')

# Resolution — lower = faster
# nside=32 (12k px): ~3 min on CPU
# nside=64 (49k px): ~30 min on CPU
# nside=128 (196k px): needs GPU
CI_MODE = os.environ.get("CI", "").lower() in ("true", "1")
NSIDE = 16 if CI_MODE else 32
NSTEPS = 20 if CI_MODE else 300

# %% [markdown]
# ## 1. Load Copernicus Marine SST
#
# The L3S product provides satellite observations of sea surface
# temperature with cloud gaps — real measurements where the sky is
# clear, missing values where clouds block the satellite's view.

# %%
print("Loading Copernicus Marine SST (L3S)...")
L3S = xr.open_zarr(
    "https://s3.waw3-1.cloudferro.com/mdl-arco-time-045/arco/"
    "SST_GLO_SST_L3S_NRT_OBSERVATIONS_010_010/"
    "IFREMER-GLOB-SST-L3-NRT-OBS_FULL_TIME_SERIE_202211/timeChunked.zarr"
).sel(time=TIME_SLICE).isel(time=0)

# Quality filter and extract SST
sst = L3S['sea_surface_temperature'].where(L3S['quality_level'] == 5).compute()
cloud_frac = (1 - sst.count().item() / sst.size) * 100
print(f"  Shape: {sst.shape}")
print(f"  Cloud fraction: {cloud_frac:.0f}%")

# %% [markdown]
# ## 2. Convert to HEALPix
#
# The scattering transform works on the sphere. We regrid the lat-lon
# SST to a HEALPix grid, which provides equal-area pixels — essential
# for unbiased statistical analysis on a sphere.

# %%
npix = 12 * NSIDE**2
print(f"Converting to HEALPix nside={NSIDE} ({npix:,} pixels)...")

lat, lon = sst.latitude.values, sst.longitude.values
lon_g, lat_g = np.meshgrid(lon, lat)
cids = hp.ang2pix(NSIDE, np.deg2rad(90 - lat_g.ravel()),
                  np.deg2rad(lon_g.ravel()), nest=True)
sst_flat = sst.values.ravel()

# Average SST per HEALPix cell
sst_hp = np.full(npix, np.nan)
cnt = np.zeros(npix)
has_data = np.zeros(npix, dtype=bool)

for i in range(len(cids)):
    has_data[cids[i]] = True
    if not np.isnan(sst_flat[i]):
        if np.isnan(sst_hp[cids[i]]):
            sst_hp[cids[i]] = sst_flat[i]
            cnt[cids[i]] = 1
        else:
            sst_hp[cids[i]] += sst_flat[i]
            cnt[cids[i]] += 1

sst_hp[cnt > 0] /= cnt[cnt > 0]

ocean = has_data
observed = cnt > 0
clouds = ocean & ~observed

# Fill cloudy pixels with mean SST as initial guess
mean_sst = np.nanmean(sst_hp[observed])
sst_filled = np.zeros(npix, dtype=np.float32)
sst_filled[observed] = sst_hp[observed]
sst_filled[clouds] = mean_sst

print(f"  Ocean cells: {ocean.sum():,}")
print(f"  Observed (clear sky): {observed.sum():,}")
print(f"  Cloudy (to fill): {clouds.sum():,}")

# %% [markdown]
# ## 3. Compute reference scattering coefficients
#
# The scattering transform captures multi-scale statistical properties
# of the SST field — its textures, gradients, fronts, and eddies.
# We compute these from the **observed (cloud-free) regions only**.
# The synthesis will then adjust the cloudy regions until the whole
# map's statistics match those of the observed data.

# %%
print("Computing scattering coefficients from observed SST...")
scat = sc.funct(NORIENT=4, KERNELSZ=3, all_type='float32', silent=True)
print(f"  Device: {scat.backend.device}")

data = sst_filled.reshape(1, npix)
mask_observed = observed.reshape(1, npix).astype(np.float32)
mask_ocean = ocean.reshape(1, npix).astype(np.float32)

ref, sref = scat.eval(data, mask=mask_observed, calc_var=True)
print("  Reference scattering coefficients computed")

# %% [markdown]
# ## 4. Define loss function and run synthesis
#
# FOSCAT starts from the initial guess (mean SST in cloudy pixels)
# and iteratively adjusts those pixels until the scattering statistics
# of the full map match those of the observed regions.
#
# Only the cloudy pixels are modified — observed values are preserved.

# %%
def The_loss(x, scat_operator, args):
    ref = args[0]
    mask = args[1]
    sref = args[2]
    learn = scat_operator.eval(x, mask=mask)
    loss = scat_operator.reduce_mean(scat_operator.square((ref - learn) / sref))
    return loss

mask_const = scat.backend.constant(scat.backend.bk_cast(mask_ocean))
loss = synthe.Loss(The_loss, scat, ref, mask_const, sref)
sy = synthe.Synthesis([loss])

mask_clouds = scat.backend.bk_cast(clouds.reshape(1, npix).astype(np.float32))

print(f"Running FOSCAT synthesis: {NSTEPS} steps...")
t0 = time.time()

omap = sy.run(
    scat.backend.bk_cast(data),
    EVAL_FREQUENCY=max(NSTEPS // 10, 1),
    grd_mask=mask_clouds,
    NUM_EPOCHS=NSTEPS,
    do_lbfgs=True
)

elapsed = time.time() - t0
omap_np = np.array(omap) if not hasattr(omap, 'numpy') else omap.numpy()
print(f"  Completed in {elapsed:.1f}s")

# %% [markdown]
# ## 5. Validate results
#
# We compare the gap-filled map against the original observations
# and check that the filled regions have physically plausible SST values.

# %%
filled_values = omap_np[0][clouds]
observed_values = omap_np[0][observed]

print("=== Validation ===")
print(f"  {'':25s} {'Observed':>12s} {'Filled clouds':>14s}")
print(f"  {'Mean (°C)':25s} {observed_values.mean():12.2f} {filled_values.mean():14.2f}")
print(f"  {'Std (°C)':25s} {observed_values.std():12.2f} {filled_values.std():14.2f}")
print(f"  {'Min (°C)':25s} {observed_values.min():12.2f} {filled_values.min():14.2f}")
print(f"  {'Max (°C)':25s} {observed_values.max():12.2f} {filled_values.max():14.2f}")

# Change from initial guess
diff = np.abs(sst_filled - omap_np[0])
changed = diff[clouds]
print(f"\n  Cloudy pixels modified: {(changed > 0.01).sum():,} / {clouds.sum():,}")
print(f"  Mean change from initial guess: {changed.mean():.2f} °C")

# Scattering coefficient comparison
out_scat = scat.eval(omap_np, mask=mask_ocean)
start_scat = scat.eval(data, mask=mask_ocean)

ref_s1 = scat.backend.to_numpy(ref.S1)
out_s1 = scat.backend.to_numpy(out_scat.S1)
start_s1 = scat.backend.to_numpy(start_scat.S1)

scat_err_start = np.mean((ref_s1 - start_s1)**2)
scat_err_out = np.mean((ref_s1 - out_s1)**2)
improvement = (1 - scat_err_out / (scat_err_start + 1e-20)) * 100

print(f"\n  Scattering coefficient error (start): {scat_err_start:.6f}")
print(f"  Scattering coefficient error (filled): {scat_err_out:.6f}")
print(f"  Improvement: {improvement:.1f}%")

# %% [markdown]
# ## 6. Save results

# %%
results = {
    "method": "Cross Scattering Transform (FOSCAT)",
    "original_paper": "Delouis et al. 2022, A&A 668, A122",
    "original_paper_doi": "10.1051/0004-6361/202244566",
    "eo_notebook_author": "Jean-Marc Delouis",
    "eo_notebook_url": "https://pangeo-data.github.io/egi2024-demo/SST_AI.html",
    "data": "Copernicus Marine SST L3S (IFREMER)",
    "date": "2024-06-01",
    "healpix_nside": NSIDE,
    "foscat_epochs": NSTEPS,
    "elapsed_seconds": elapsed,
    "device": str(scat.backend.device),
    "ocean_cells": int(ocean.sum()),
    "observed_cells": int(observed.sum()),
    "cloudy_cells": int(clouds.sum()),
    "cloud_fraction_pct": float(cloud_frac),
    "results": {
        "observed_mean": float(observed_values.mean()),
        "filled_mean": float(filled_values.mean()),
        "observed_std": float(observed_values.std()),
        "filled_std": float(filled_values.std()),
        "mean_change_degC": float(changed.mean()),
        "scattering_improvement_pct": float(improvement),
    },
}

with open(RESULTS / "sst_gap_filling_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {RESULTS / 'sst_gap_filling_results.json'}")

# %%
fig = plt.figure(figsize=(18, 10))

# Row 1: Maps
hp.mollview(sst_filled * (ocean.astype(float)) + (~ocean) * hp.UNSEEN,
            nest=True, min=0, max=30, cmap='coolwarm',
            title='SST with cloud gaps (initial guess = mean)',
            sub=(2, 3, 1), fig=fig.number)
hp.mollview(omap_np[0] * (ocean.astype(float)) + (~ocean) * hp.UNSEEN,
            nest=True, min=0, max=30, cmap='coolwarm',
            title=f'FOSCAT gap-filled ({NSTEPS} steps)',
            sub=(2, 3, 2), fig=fig.number)
hp.mollview(clouds.astype(float) + (~ocean) * hp.UNSEEN,
            nest=True, cmap='binary',
            title='Cloud mask',
            sub=(2, 3, 3), fig=fig.number)

# Row 2: Diagnostics
ax4 = fig.add_subplot(2, 3, 4)
ax4.hist(observed_values, bins=50, alpha=0.6, label='Observed', density=True)
ax4.hist(filled_values, bins=50, alpha=0.6, label='Filled clouds', density=True)
ax4.set_xlabel('SST (°C)')
ax4.set_ylabel('Density')
ax4.set_title('Pixel distribution')
ax4.legend()

ax5 = fig.add_subplot(2, 3, 5)
ax5.hist(changed[changed > 0.01], bins=50, color='steelblue')
ax5.set_xlabel('Change from initial guess (°C)')
ax5.set_ylabel('Count')
ax5.set_title('How much did cloudy pixels change?')

ax6 = fig.add_subplot(2, 3, 6)
history = sy.get_history()
valid = history[history > 0]
ax6.semilogy(valid)
ax6.set_xlabel('Iteration')
ax6.set_ylabel('Loss')
ax6.set_title('Synthesis convergence')

fig.suptitle('Cross Scattering Transform — Copernicus SST Gap-Filling',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(RESULTS / 'sst_gap_filling.png', dpi=150, bbox_inches='tight')
print(f"Saved: {RESULTS / 'sst_gap_filling.png'}")

# %% [markdown]
# ## 7. What does this mean?
#
# The scattering transform — originally developed for Planck astrophysics
# data — fills cloud gaps in Copernicus SST satellite imagery. The filled
# regions have physically plausible temperature values that preserve the
# spatial structure and statistical properties of the observed ocean
# temperature field.
#
# This demonstrates that mathematical methods developed for one imaging
# domain (astrophysics) transfer to another (Earth observation) when
# the underlying structure is similar — both are images on a sphere
# with multi-scale spatial correlations.
#
# ## Companion projects
#
# - **Astrophysics reproduction**: [fiesta-scattering-astro](https://github.com/annefou/fiesta-scattering-astro)
# - **Biodiversity application**: fiesta-scattering-bio (coming soon)
#
# ## Replication context
#
# This is a reproduction of Jean-Marc Delouis's EO application notebook,
# part of the [FIESTA-OSCARS](https://oscars-project.eu) project
# demonstrating cross-domain FAIR image analysis workflows.
# Published as FORRT nanopublications on
# [Science Live](https://platform.sciencelive4all.org).
