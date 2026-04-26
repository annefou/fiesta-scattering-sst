# FIESTA Scattering SST: Copernicus Sea Surface Temperature Gap-Filling

[![Source DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19686691.svg)](https://doi.org/10.5281/zenodo.19686691)
[![Docker image DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19708070.svg)](https://doi.org/10.5281/zenodo.19708070)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
[OSCARS](https://oscars-project.eu/projects/fair-image-analysis-across-sciences) programme, demonstrating cross-domain
reproducibility of research software methods.

## FORRT nanopublication chain

The full provenance of this replication is recorded as a six-step FORRT
nanopublication chain on the
[Science Live](https://platform.sciencelive4all.org) platform. Each step is
independently citable and machine-readable; together they form the FAIR
provenance receipt for this replication.

> **Headline assertion — machine-readable:**
> [**This replication `cito:confirms` + `cito:usesMethodIn` Delouis et al. 2022, AND `cito:credits` the IGARSS 2024 Pangeo tutorial**](https://w3id.org/sciencelive/np/RAA6_hyyQvv4h0l4bFNt-9tEn2KRRIL4mVE0pRzt4W31k)
>
> The CiTO citation nanopublication encodes three relationships at once:
> we substantiate the paper's generalisation claim across domains
> (`cito:confirms` Delouis 2022); our work uses the FOSCAT scattering
> method developed in that paper (`cito:usesMethodIn` Delouis 2022); and
> the operational SST workflow we follow is Jean-Marc Delouis's IGARSS
> 2024 Pangeo tutorial notebook (`cito:credits`). Discovery tools
> (Scholia, Wikidata pipelines, SPARQL endpoints) can follow this single
> citation to find all three relationships.

The five preceding nanopubs build the provenance ladder up to that citation:

| Step | Type | Asserts | Nanopub URI |
|---|---|---|---|
| 1 | Quote-with-comment (Annotate a paper quotation) | Verbatim quote of Delouis et al. 2022's generalisation claim (Section 6 Conclusion), with personal comment on the cross-domain test | [`RAUqG…`](https://w3id.org/sciencelive/np/RAUqGWdJOve1i0KCOCZhPfXT9qXquiS3qFdLNLgKu-j-I) |
| 2 | AIDA sentence | Atomic, declarative restatement: scattering transforms can fill cloud gaps in satellite SST observations using a gap-free reference product as a statistical target. *(Published via Nanodash because of a Science Live AIDA-form bug with the datasets + publications fields; URI is on the bare-`np` namespace rather than `sciencelive/np`.)* | [`RAsnO…`](https://w3id.org/np/RAsnOdj5BxkhD_u67TguhaVIz6T1TaIGW5qzM7_QuNrqU) |
| 3 | FORRT Claim (model performance) | The SST gap-filling claim, typed as a FORRT model-performance claim | [`RAQPv…`](https://w3id.org/sciencelive/np/RAQPvE7Y4PNeL2oDwFh_uJgJbFHyBmWEvyZfO-RWy1pP8) |
| 4 | FORRT Reproduction/Replication Study | Both reproduction (of Jean-Marc's IGARSS 2024 SST notebook) AND replication (of Delouis 2022's generalisation claim across domains) — same FOSCAT software, Earth observation data instead of astrophysics | [`RA45t…`](https://w3id.org/sciencelive/np/RA45t1bdfz6Jr40G9dDqrWAF4i7-DT3HQNOSheRWjcuho) |
| 5 | FORRT Replication Outcome (Validated, High) | FOSCAT RMSE 0.989 K vs L4 reference; harmonic baseline RMSE 11.46 K; 91% improvement | [`RAK2y…`](https://w3id.org/sciencelive/np/RAK2ynlqgA3L_YVbEo-cVoXZ7T63q4eCFZqf9CPWWNc40) |
| 6 | **CiTO citation — `cito:confirms` + `cito:usesMethodIn` Delouis 2022 + `cito:credits` IGARSS 2024 tutorial** | The headline triple assertion above | [**`RAA6_…`**](https://w3id.org/sciencelive/np/RAA6_hyyQvv4h0l4bFNt-9tEn2KRRIL4mVE0pRzt4W31k) |

The chain runs: paper → quote → atomic claim → FORRT claim → study (this
repo) → outcome (the metrics in the validation table) → CiTO citations
back to the paper *and* to Jean-Marc's IGARSS 2024 tutorial.

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

## Note on FOSCAT and GPU/CPU support

The [FOSCAT](https://github.com/jmdelouis/FOSCAT) package (as of v2026.2.7 on
PyPI) has several hardcoded `device='cuda'` defaults, which means it **only
works on machines with an NVIDIA GPU** out of the box. On CPU-only machines
(Apple Silicon Macs, CI runners, etc.) it will crash with a CUDA device error.

We have submitted a fix upstream:
[jmdelouis/FOSCAT#40](https://github.com/jmdelouis/FOSCAT/pull/40)
([commit](https://github.com/annefou/FOSCAT/commit/04244ed)).

Until the fix is merged and released, you can install FOSCAT from our fork:

```bash
pip install git+https://github.com/annefou/FOSCAT.git@v0.1.0-cpu
```

The fix is fully backwards compatible: on CUDA machines the behaviour is
identical to the original. It simply adds auto-detection so that CPU is used as
a fallback when CUDA is not available.

## Container image

A Docker container is built on every release and pushed to GitHub Container
Registry, and archived to Zenodo for long-term preservation.

```bash
docker pull ghcr.io/annefou/fiesta-scattering-sst:latest
docker run --rm -v "$PWD/results:/app/results" \
    -e COPERNICUSMARINE_SERVICE_USERNAME=... \
    -e COPERNICUSMARINE_SERVICE_PASSWORD=... \
    ghcr.io/annefou/fiesta-scattering-sst:latest
```

Zenodo-archived tarballs of every released image are available via the
[Docker image concept DOI 10.5281/zenodo.19708070](https://doi.org/10.5281/zenodo.19708070).

## How to cite

If you use this repository, please cite it via its Zenodo DOI together with
the original method paper (Delouis et al. 2022).

```
Fouilloux, A. (2026). FIESTA Scattering SST: Copernicus Sea Surface
Temperature Gap-Filling (v0.3.1). Zenodo.
https://doi.org/10.5281/zenodo.19686691
```

BibTeX:

```bibtex
@software{fouilloux_fiesta_scattering_sst,
  author    = {Fouilloux, Anne},
  title     = {FIESTA Scattering SST: Copernicus Sea Surface Temperature Gap-Filling},
  year      = {2026},
  version   = {0.3.1},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19686691},
  url       = {https://doi.org/10.5281/zenodo.19686691}
}
```

The DOI above is the **concept DOI** — it always resolves to the latest
release. Specific version DOIs are available on the
[Zenodo record page](https://doi.org/10.5281/zenodo.19686691).

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## Companion repository

- Astrophysics (CMB): [annefou/fiesta-scattering-astro](https://github.com/annefou/fiesta-scattering-astro)

## License

MIT --- see [LICENSE](LICENSE).

## Author

Anne Fouilloux, LifeWatch ERIC
([ORCID 0000-0002-1784-2920](https://orcid.org/0000-0002-1784-2920))
