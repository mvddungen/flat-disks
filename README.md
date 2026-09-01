# flat-disks

This repository contains the code developed for the Master's thesis

**JT Gravity and the Continuum Limit of Random Flat Disks**

by Marieke van den Dungen at Radboud University, Nijmegen, the Netherlands (2026).

The code implements the numerical experiments used to study uniform random flat disks on the square lattice and their proposed continuum description in terms of Gaussian disk sectors.

## Repository overview

The repository is organized by thesis chapter, with separate directories for the numerical disk mappings, the Fourier analysis of the conformal factor, and the Gaussian disk-sector comparison. Each directory contains the corresponding notebook and, where applicable, the processed data used in the analysis. 

## Disk mappings

`ch3_disk_mappings/disk_mappings.ipynb`

This notebook contains the numerical construction of conformal maps between square-lattice disks and the unit disk.

For a given lattice disk, the notebook:

- reads the lattice geometry and ordered boundary
- constructs the discrete graph Laplacian
- computes the harmonic measure of the boundary
- assigns boundary preimages on the unit circle
- solves the discrete Dirichlet problem to obtain interior preimages
- fits a polynomial approximation to the conformal map
- visualizes the resulting mapping

The files in `ch3_disk_mappings/data/example_disks/` contain example lattice disks of different sizes.

## Fourier analysis of the conformal factor

`ch3_fourier_analysis/disk_mapping_fourier.ipynb`

This notebook analyzes ensembles of polynomial approximations to the conformal maps for varying fitting radii and disk sizes.

In particular, it studies the conformal factor

$$
\sigma(\theta) = \log |F'(e^{i\theta})|
$$

through its Fourier modes. The numerical experiments investigate the dependence of the fitted conformal-map coefficients and Fourier modes on the disk size and effective fitting radius.

The Fourier-mode power is compared with the scaling expected for a log-correlated Gaussian free field.

The fitted polynomial coefficients used in this analysis are stored as HDF5 files in `ch3_fourier_analysis/data/`. The files correspond to effective fitting radii

$$
r \in \{0.6, 0.7, 0.8, 0.9\}
$$

and lattice-disk perimeters

$$
m \in \{50, 100, 200, 500, 1000, 2000\}.
$$

## Gaussian disk sectors

`ch5_gaussian_sectors/gaussian_comparison.ipynb`

This notebook contains the numerical comparison between the random flat-disk observables and the Gaussian disk-sector model.

The Gaussian field is approximated by a finite random series with truncation degrees

$$
N \in \{100, 500, 1000, 5000\}.
$$

The finite-mode approximations are generated from a truncated random analytic series with independent real Gaussian coefficients.

For each realization, a radial distance is determined numerically by solving the differential equation for the preimage of a radial segment.

Two statistical predictions are tested:

1. the expected length of the radial cut
2. the probability distribution of the boundary-length ratio

The notebook also contains the numerical inverse Mellin transform used to evaluate the corresponding prediction of the discrete disk model.

The processed data used for these comparisons are stored in `ch5_gaussian_sectors/data/rplt_data.hdf5`.

## Data availability

The processed numerical data required to reproduce the analyses and figures are included in this repository.
The full ensemble files used in the numerical experiments are not included in the public repository because of their large sizes. These are:

- ensembles of random flat disks on the square lattice 
- ensembles of finite-mode Gaussian disk sector approximations

The generation code for both ensembles can be requested from [Timothy Budd](https://hef.ru.nl/~tbudd/) at Radboud University.

## Requirements

The notebooks require Python 3.11 or later and use the following main packages:

- NumPy
- SciPy
- Matplotlib
- h5py
- ipykernel

The dependencies are specified in `pyproject.toml`.
