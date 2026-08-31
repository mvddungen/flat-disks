# flat-disks
This repository includes all code written for the Masters thesis titled 

**JT Gravity and the Continuum Limit of Random Flat Disks**

by Marieke van den Dungen at Radboud University, Nijmegen, the Netherlands. The code implements the numerical experiments used to study uniform random flat disks, and their proposed continuum description in terms of Gaussian disk sectors.

## Repository structure

```text
flat-disks/
│
├── ch3_disk_mappings/
│   ├── data/
│   │   ├── latticediskh10.txt
│   │   ├── latticediskh100.txt
│   │   ├── latticediskh1000.txt
│   │   └── latticediskh10000.txt
│   └── disk_mappings.ipynb
│
├── ch3_fourier_analysis/
│   ├── data/
│   └── disk_mapping_fourier.ipynb
│
├── ch5_gaussian_sectors/
│   ├── data/
│   │   ├── rplt_data.hdf5
│   │   ├── samples_N100
│   │   ├── samples_N500
│   │   ├── samples_N1000
│   │   └── samples_N5000
│   └── gaussian_comparison.ipynb
│
├── pyproject.toml
```

## Disk mappings

`ch3_disk_mappings/disk_mappings.ipynb`

This notebook contains the numerical construction of conformal maps between square-lattice disks and the unit disk.

For a given lattice disk, the notebook:

- reads the lattice geometry and ordered boundary
- constructs the discrete graph Laplacian
- computes the harmonic measure of the boundary
- assigns boundary preimages on the unit circle
- solves the discrete Dirichlet problem to obtain interior preimages
- fits a polynomial approximation to the inverse conformal map
- visualizes the resulting mapping

The files in `ch3_disk_mappings/data/` contain example lattice disks of different sizes.

## Fourier analysis of conformal factor

`ch3_fourier_analysis/disk_mapping_fourier.ipynb`

This notebook analyzes ensembles of polynomial approximations to the conformal maps for varying fitting radii and disk sizes.

In particular, it studies the conformal factor

$$
\sigma(\theta) = \log |F'(e^{i\theta})|
$$

through its Fourier modes. The numerical experiments investigate the dependence of the fitted conformal map coefficients and Fourier modes on the disk size and effective fitting radius.

The Fourier-mode power is compared with the scaling expected for a log-correlated Gaussian free field.

The required coefficient data of the polynomial approximations are stored in `ch3_fourier_analysis/data/`.

## Gaussian disk sectors

`ch5_gaussian_sectors/gaussian_comparison.ipynb`

This notebook contains the numerical comparison between the random flat-disk observables and the Gaussian disk-sector model.

Gaussian fields are approximated by finite polynomial expansions with degrees

$$
N \in \{100, 500, 1000, 5000\}.
$$

For each realization, a radial distance is determined numerically by solving the differential equation for the preimage of a radial segment.

Two statistical predictions are tested:

1. the expected length of the radial cut
2. the probability distribution of the boundary-length ratio

The notebook also contains the numerical inverse Mellin transform used to evaluate the corresponding prediction of the discrete disk model.

The Gaussian samples and computed radial distances are stored in `ch5_gaussian_sectors/data/`.

## Requirements

The notebooks are written in Python and use the following main packages:

- NumPy
- SciPy
- Matplotlib
- h5py

