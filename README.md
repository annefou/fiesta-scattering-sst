# FIESTA Scattering SST: Copernicus Sea Surface Temperature Gap-Filling

Reproducing the application of **Cross Scattering Transform** to fill cloud gaps in
Copernicus Marine Sea Surface Temperature (SST) satellite data.

## Background

Satellite-derived SST imagery from the Copernicus Marine Service suffers from
persistent cloud-cover gaps --- clouds block the infrared sensors, leaving large
missing regions in daily composites. Filling these gaps accurately is essential
for climate monitoring, ocean modelling, and marine ecosystem studies.

**Jean-Marc Delouis** developed the Cross Scattering Transform method in the
context of astrophysics (CMB component separation) and subsequently demonstrated
its effectiveness for Earth Observation, applying it to Copernicus Marine SST
cloud gap-filling. This repository reproduces his EO application.

### Cross-domain story: astrophysics to Earth Observation

The scattering transform was originally designed for analysing the Cosmic
Microwave Background (CMB) on the sphere (HEALPix grids). Delouis et al. (2022)
showed that cross-scattering statistics capture non-Gaussian correlations between
fields at different scales, enabling robust component separation and in-painting.

The same mathematical framework transfers directly to ocean remote sensing:
replace CMB temperature maps with SST fields, replace foreground contamination
with cloud masks, and the scattering-based synthesis fills the gaps while
preserving the multi-scale structure of ocean features (eddies, fronts,
upwelling patterns).

This SST repository is the **Earth Observation companion** to the astrophysics
reproduction at [annefou/fiesta-scattering-astro](https://github.com/annefou/fiesta-scattering-astro).

## Credits

- **Method and EO application:** Jean-Marc Delouis
- **Reference:** Delouis, Allys, Gauvrit & Boulanger (2022),
  *Cross-scattering transform on the sphere*, A&A 668, A122.
  [DOI: 10.1051/0004-6361/202244566](https://doi.org/10.1051/0004-6361/202244566)
- **Software:** [foscat](https://github.com/jmdelouis/FOSCAT) (Forward Scattering Transform)

## FIESTA-OSCARS

This work is part of the **FIESTA** project under the
[OSCARS](https://oscars-project.eu/) programme, demonstrating cross-domain
reproducibility of research software methods.

## Data

SST data is downloaded automatically from the
[Copernicus Marine Service](https://marine.copernicus.eu/) (L3S product).
No credentials are needed for the L3S dataset.

## Quick start

### Conda / Mamba

```bash
mamba env create -f environment.yml
mamba activate fiesta-scattering-sst
python 01_sst_gap_filling.py
```

### Docker (GPU)

```bash
docker build -t fiesta-sst .
docker run --gpus all fiesta-sst
```

### Snakemake

```bash
snakemake --cores 1
```

### Jupyter Book

```bash
npm install mystmd
npx myst build --html
```

## Companion repository

- Astrophysics (CMB): [annefou/fiesta-scattering-astro](https://github.com/annefou/fiesta-scattering-astro)

## License

MIT --- see [LICENSE](LICENSE).

## Author

Anne Fouilloux, LifeWatch ERIC
([ORCID 0000-0002-1784-2920](https://orcid.org/0000-0002-1784-2920))
