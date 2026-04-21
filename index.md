# Scattering Transform: SST Gap-Filling

## The problem: cloud gaps in satellite SST

Sea Surface Temperature (SST) is one of the most important Essential Climate
Variables. The Copernicus Marine Service provides daily SST composites derived
from infrared satellite sensors. However, clouds are opaque to infrared
radiation, leaving large gaps in every daily image --- sometimes obscuring more
than half the ocean surface in a region.

Traditional gap-filling approaches (optimal interpolation, EOF reconstruction)
struggle to preserve the fine-scale ocean structures --- mesoscale eddies,
thermal fronts, coastal upwelling filaments --- that make satellite SST
scientifically valuable.

## The solution: Cross Scattering Transform

The **scattering transform** decomposes a signal into a hierarchy of wavelet
modulus coefficients that capture information at progressively finer scales,
including non-Gaussian features that power spectra miss. The **cross-scattering
transform** extends this to pairs of fields, learning the statistical
relationships between them across scales.

For SST gap-filling, the method:

1. Computes cross-scattering statistics between the incomplete SST field and
   auxiliary fields (e.g. previous days, climatology).
2. Synthesises a gap-filled SST field whose scattering statistics match the
   target, effectively "painting in" the missing regions with physically
   consistent ocean structure.

The approach was developed by **Jean-Marc Delouis** and collaborators, first for
astrophysical component separation on the sphere (CMB foreground removal), then
applied by Delouis to Earth Observation SST data.

## Cross-domain connection: astrophysics to Earth Observation

This work demonstrates a powerful cross-domain transfer:

- **Astrophysics origin:** The scattering transform was designed for analysing
  Cosmic Microwave Background (CMB) maps on HEALPix spherical grids. Delouis
  et al. (2022) used it for CMB component separation --- removing Galactic dust
  and synchrotron foregrounds from the cosmological signal.
- **Earth Observation application:** The same mathematics applies to ocean
  remote sensing. Replace CMB temperature fluctuations with SST fields, replace
  Galactic foreground masks with cloud masks, and the scattering-based synthesis
  fills the gaps while preserving multi-scale ocean features.

The companion astrophysics repository reproduces the original CMB application:
[annefou/fiesta-scattering-astro](https://github.com/annefou/fiesta-scattering-astro).

## Data source

SST data is obtained from the
[Copernicus Marine Service](https://marine.copernicus.eu/) L3S product
and is downloaded automatically --- no credentials are required.

## Reference

> Delouis, Allys, Gauvrit & Boulanger (2022), *Cross-scattering transform on
> the sphere*, Astronomy & Astrophysics, 668, A122.
> [DOI: 10.1051/0004-6361/202244566](https://doi.org/10.1051/0004-6361/202244566)
