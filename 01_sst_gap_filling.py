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
# Copernicus Marine SST data.
#
# ## Data access
#
# ### Prerequisites
#
# You need a **free Copernicus Marine account**:
#
# 1. Register at https://data.marine.copernicus.eu/register
# 2. Install: `pip install copernicusmarine`
# 3. Login: `copernicusmarine login`
#
# ## Credits
#
# - **Method**: Jean-Marc Delouis, LOPS/CNRS
# - **Original paper**: Delouis et al. (2022), A&A 668, A122
# - **This reproduction**: Anne Fouilloux, LifeWatch ERIC (FIESTA-OSCARS)

# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
import healpy as hp
import foscat.scat_cov as sc
import foscat.Synthesis as synthe
import copernicusmarine
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

DATE = "2026-04-01"

CI_MODE = os.environ.get("CI", "").lower() in ("true", "1")
NSIDE = 16 if CI_MODE else 32
NSTEPS = 20 if CI_MODE else 300
LMAX = 15 if CI_MODE else 30

# %% [markdown]
# ## 1. Load SST data
#
# We use the **PMW L3S** product (0.25° resolution) and the **L4**
# product (0.25° resolution) — both at the same grid, avoiding
# resolution mismatch issues.
#
# - **L3S PMW**: passive microwave SST with cloud gaps
# - **L4**: gap-free analysis (ground truth for validation)

# %%
print(f"Loading Copernicus Marine SST for {DATE}...")

# L3S PMW: 0.25° resolution — matches L4
print("  L3S PMW (0.25°, with cloud gaps)...")
L3S = copernicusmarine.open_dataset(
    dataset_id="cmems_obs-sst_glo_phy_l3s_pmw_P1D-m",
    variables=["sea_surface_temperature", "quality_level"],
    start_datetime=DATE, end_datetime=DATE,
).isel(time=0)
print(f"    Shape: {L3S['sea_surface_temperature'].shape}")

# L4: 0.25° resolution — gap-free reference
print("  L4 (0.25°, gap-free)...")
L4 = copernicusmarine.open_dataset(
    dataset_id="cmems_obs-sst_glo_phy-temp_nrt_P1D-m",
    variables=["analysed_sst"],
    start_datetime=DATE, end_datetime=DATE,
).isel(time=0)
print(f"    Shape: {L4['analysed_sst'].shape}")

# %% [markdown]
# ## 2. Preprocess on the same lat-lon grid
#
# Following Jean-Marc's approach: create a single dataset with both
# L3S and L4 on the same grid, with a proper ocean mask.

# %%
# Create ocean mask from L4 (NaN = land/ice)
sst_l4 = L4['analysed_sst'].compute()
mask = ~sst_l4.isnull()
print(f"Ocean mask: {mask.sum().item():,} pixels")

# Create unified dataset on L4's grid
ds = xr.Dataset()
ds['mask'] = mask
ds['SST_L4'] = sst_l4

# L3S: quality filter, then regrid to L4 grid if needed
sst_l3s = L3S['sea_surface_temperature'].where(
    L3S['quality_level'] == 5
).compute()

# Regrid L3S to L4 grid (interp to matching lat/lon)
if sst_l3s.shape != sst_l4.shape:
    print(f"  Regridding L3S {sst_l3s.shape} to L4 {sst_l4.shape}...")
    sst_l3s = sst_l3s.interp(
        latitude=sst_l4.latitude, longitude=sst_l4.longitude,
        method='nearest'
    )

# Mask land/ice with -100 (like Jean-Marc does — not NaN, to distinguish from clouds)
ds['SST_L3S'] = sst_l3s.where(mask, -100)

n_valid = (ds['SST_L3S'] > 0).sum().item()
n_ocean = mask.sum().item()
cloud_frac = (1 - n_valid / n_ocean) * 100
print(f"Cloud fraction: {cloud_frac:.0f}%")
print(f"L3S range: {float(ds['SST_L3S'].where(ds['SST_L3S'] > 0).min()):.1f} to "
      f"{float(ds['SST_L3S'].where(ds['SST_L3S'] > 0).max()):.1f}")
print(f"L4 range:  {float(ds['SST_L4'].where(mask).min()):.1f} to "
      f"{float(ds['SST_L4'].where(mask).max()):.1f}")

# %% [markdown]
# ## 3. Convert to HEALPix
#
# Both L3S and L4 are converted to HEALPix together, from the same
# unified dataset on the same grid.

# %%
npix = 12 * NSIDE**2
print(f"Converting to HEALPix nside={NSIDE} ({npix:,} pixels)...")

lat, lon = ds.latitude.values, ds.longitude.values
lon_g, lat_g = np.meshgrid(lon, lat)
th = np.deg2rad(90 - lat_g.ravel())
ph = np.deg2rad(lon_g.ravel())
cids = hp.ang2pix(NSIDE, th, ph, nest=True)

# Average per HEALPix cell
def to_healpix(data_2d, cell_ids, npix, fill_val=np.nan):
    flat = data_2d.ravel()
    hp_sum = np.zeros(npix)
    hp_cnt = np.zeros(npix)
    for i in range(len(cell_ids)):
        v = flat[i]
        if not np.isnan(v) and v > -90:  # skip NaN and -100 masked values
            hp_sum[cell_ids[i]] += v
            hp_cnt[cell_ids[i]] += 1
    result = np.full(npix, fill_val)
    valid = hp_cnt > 0
    result[valid] = hp_sum[valid] / hp_cnt[valid]
    return result, valid

# Ocean mask on HEALPix
mask_flat = mask.values.astype(float)
mask_hp_sum = np.zeros(npix)
mask_hp_cnt = np.zeros(npix)
for i in range(len(cids)):
    mask_hp_sum[cids[i]] += mask_flat.ravel()[i]
    mask_hp_cnt[cids[i]] += 1
ocean_hp = (mask_hp_sum / np.maximum(mask_hp_cnt, 1)) > 0.5

# L3S and L4 on HEALPix
sst_l3s_hp, observed = to_healpix(ds['SST_L3S'].values, cids, npix)
sst_l4_hp, _ = to_healpix(ds['SST_L4'].values, cids, npix)

clouds = ocean_hp & ~observed

# Fill L4 non-ocean with ocean mean
ocean_mean_l4 = np.nanmean(sst_l4_hp[ocean_hp])
sst_l4_hp[~ocean_hp | np.isnan(sst_l4_hp)] = ocean_mean_l4

print(f"  Ocean: {ocean_hp.sum():,}, Observed: {observed.sum():,}, Cloudy: {clouds.sum():,}")

# %% [markdown]
# ## 4. Spherical harmonics baseline

# %%
print(f"Spherical harmonics baseline (lmax={LMAX})...")
l, m = hp.Alm.getlm(lmax=LMAX)
n_alm = (m == 0).sum() + 2 * (m > 0).sum()
function = np.zeros([n_alm, npix])
alm = np.zeros([l.shape[0]], dtype='complex')
i = 0
for k in range(l.shape[0]):
    alm[:] = 0; alm[k] = 1
    function[i] = hp.alm2map(alm, NSIDE, verbose=False); i += 1
    if m[k] > 0:
        alm[k] = 1j
        function[i] = hp.alm2map(alm, NSIDE, verbose=False); i += 1

obs_idx = np.where(observed)[0]
mat = function[:, obs_idx] @ function[:, obs_idx].T
vec = function[:, obs_idx] @ sst_l3s_hp[obs_idx]
coef = np.linalg.solve(mat, vec)
fit_data = coef @ function

sst_polyfit = np.zeros(npix, dtype=np.float32)
sst_polyfit[observed] = sst_l3s_hp[observed]
sst_polyfit[clouds] = fit_data[clouds]
sst_polyfit[~ocean_hp] = ocean_mean_l4

print(f"  Filled {clouds.sum():,} cloudy cells")

# %% [markdown]
# ## 5. FOSCAT gap-filling with L4 reference
#
# The key: FOSCAT uses the **L4 gap-free product** as statistical
# reference. It learns what the complete SST field looks like, then
# adjusts cloudy pixels to match those multi-scale texture statistics.

# %%
print("FOSCAT synthesis...")
scat = sc.funct(NORIENT=4, KERNELSZ=3, all_type='float32', silent=True)
print(f"  Device: {scat.backend.device}")

data = sst_polyfit.reshape(1, npix).astype(np.float32)
mask_ocean = ocean_hp.reshape(1, npix).astype(np.float32)

# Reference from L4 gap-free product
ref, sref = scat.eval(
    sst_l4_hp.reshape(1, npix).astype(np.float32),
    mask=mask_ocean, calc_var=True
)

def The_loss(x, scat_operator, args):
    ref, mask, sref = args[0], args[1], args[2]
    learn = scat_operator.eval(x, mask=mask)
    loss = scat_operator.reduce_mean(scat_operator.square((ref - learn) / sref))
    return loss

mask_const = scat.backend.constant(scat.backend.bk_cast(mask_ocean))
loss = synthe.Loss(The_loss, scat, ref, mask_const, sref)
sy = synthe.Synthesis([loss])

mask_clouds_t = scat.backend.bk_cast(clouds.reshape(1, npix).astype(np.float32))

print(f"  Running {NSTEPS} steps...")
t0 = time.time()
omap = sy.run(scat.backend.bk_cast(data), EVAL_FREQUENCY=max(NSTEPS//10, 1),
              grd_mask=mask_clouds_t, NUM_EPOCHS=NSTEPS, do_lbfgs=True)
elapsed = time.time() - t0
omap_np = np.array(omap) if not hasattr(omap, 'numpy') else omap.numpy()
print(f"  Done in {elapsed:.0f}s")

# %% [markdown]
# ## 6. Validation

# %%
diff_harm = sst_l4_hp[clouds] - sst_polyfit[clouds]
diff_foscat = sst_l4_hp[clouds] - omap_np[0][clouds]

rmse_harm = np.sqrt(np.mean(diff_harm**2))
rmse_foscat = np.sqrt(np.mean(diff_foscat**2))

print("=== Validation vs L4 (cloudy regions) ===")
print(f"  Harmonics RMSE: {rmse_harm:.3f} K")
print(f"  FOSCAT RMSE:    {rmse_foscat:.3f} K")
print(f"  Improvement:    {(1 - rmse_foscat/rmse_harm)*100:.1f}%")
print(f"  FOSCAT range:   {omap_np[0][clouds].min():.1f} to {omap_np[0][clouds].max():.1f} K")
print(f"  L4 range:       {sst_l4_hp[clouds].min():.1f} to {sst_l4_hp[clouds].max():.1f} K")

# %% [markdown]
# ## 7. Save and plot

# %%
results = {
    "method": "Cross Scattering Transform (FOSCAT)",
    "original_paper_doi": "10.1051/0004-6361/202244566",
    "l3s_dataset": "cmems_obs-sst_glo_phy_l3s_pmw_P1D-m",
    "l4_dataset": "cmems_obs-sst_glo_phy-temp_nrt_P1D-m",
    "date": DATE, "nside": NSIDE, "nsteps": NSTEPS,
    "elapsed_s": elapsed, "device": str(scat.backend.device),
    "harmonics_rmse": float(rmse_harm), "foscat_rmse": float(rmse_foscat),
    "improvement_pct": float((1 - rmse_foscat/rmse_harm) * 100),
}
with open(RESULTS / "sst_gap_filling_results.json", "w") as f:
    json.dump(results, f, indent=2)

# %%
fig = plt.figure(figsize=(18, 12))
vmin, vmax = np.percentile(sst_l4_hp[ocean_hp], [2, 98])

def plot_hp(data, mask, title, sub):
    d = data.copy(); d[~mask] = hp.UNSEEN
    hp.mollview(d, nest=True, min=vmin, max=vmax, cmap='coolwarm',
                title=title, sub=sub, fig=fig.number)

plot_hp(sst_polyfit, ocean_hp, 'L3S + harmonics fill', (3,3,1))
plot_hp(omap_np[0], ocean_hp, f'FOSCAT ({NSTEPS} steps)', (3,3,2))
plot_hp(sst_l4_hp, ocean_hp, 'L4 reference', (3,3,3))

diff_h = np.abs(sst_l4_hp - sst_polyfit); diff_h[~ocean_hp] = hp.UNSEEN
diff_f = np.abs(sst_l4_hp - omap_np[0]); diff_f[~ocean_hp] = hp.UNSEEN
hp.mollview(diff_h, nest=True, min=0, max=5, cmap='hot',
            title='|Harmonics - L4|', sub=(3,3,4), fig=fig.number)
hp.mollview(diff_f, nest=True, min=0, max=5, cmap='hot',
            title='|FOSCAT - L4|', sub=(3,3,5), fig=fig.number)
cloud_p = clouds.astype(float); cloud_p[~ocean_hp] = hp.UNSEEN
hp.mollview(cloud_p, nest=True, cmap='binary', title='Clouds', sub=(3,3,6), fig=fig.number)

ax7 = fig.add_subplot(3,3,7)
ax7.hist(diff_harm, bins=80, range=[-10,10], alpha=0.7, label=f'Harm RMSE={rmse_harm:.2f}', color='red')
ax7.hist(diff_foscat, bins=80, range=[-10,10], alpha=0.7, label=f'FOSCAT RMSE={rmse_foscat:.2f}', color='blue')
ax7.set_xlabel('Error (K)'); ax7.legend(fontsize=8); ax7.set_title('Error distribution')

ax8 = fig.add_subplot(3,3,8)
ax8.hist(omap_np[0][clouds], bins=50, alpha=0.6, label='FOSCAT', density=True)
ax8.hist(sst_l4_hp[clouds], bins=50, alpha=0.6, label='L4 truth', density=True)
ax8.set_xlabel('SST (K)'); ax8.legend(fontsize=8); ax8.set_title('Cloud distributions')

ax9 = fig.add_subplot(3,3,9)
h = sy.get_history(); ax9.semilogy(h[h>0])
ax9.set_xlabel('Iteration'); ax9.set_ylabel('Loss'); ax9.set_title('Convergence')

fig.suptitle(f'Scattering Transform SST Gap-Filling (nside={NSIDE}, {DATE})', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(RESULTS / 'sst_gap_filling.png', dpi=150, bbox_inches='tight')
print(f"Saved: {RESULTS / 'sst_gap_filling.png'}")
