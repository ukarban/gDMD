# Dynamic mode decomposition using standard particle image velocimetry data

This repository provides the reproducibility package for the paper 
**“Dynamic mode decomposition using standard particle image velocimetry data.”** 
It contains scripts for applying gap-based Dynamic Mode Decomposition (gDMD) to three datasets:

1. a confined two-dimensional cylinder wake,
2. a fluidic pinball wake with three circular cylinders, and
3. the experimental rectangular-jet PTV/PIV dataset of Neal et al. as distributed
   in the public `DingShizhe/PTV-Dataset` repository.

The DNS cases are split into two stages:

1. generate the DNS/particle-pair database;
2. construct Eulerian gap fields and compare gDMD against exact DMD.

The experimental case is split into two analogous preprocessing/analysis stages:

1. construct the mean-flow grid used for interpolation and DMD-window selection;
2. construct PTV velocity/gap fields and compare gDMD against exact DMD.

## Repository layout

```text
gdmd-examples/
├── src/gdmd/
│   ├── analysis.py         # shared exact-DMD/gDMD comparison
│   ├── dmd.py              # paired-snapshot exact/projected DMD
│   ├── dns_gap.py          # Ray-backed DNS particle-pair gap fields
│   ├── gaps.py             # second-moment particle-gap construction
│   ├── mode_matching.py    # discrete/continuous-time mode matching
│   ├── ptv.py              # experimental PTV loading/mean/grid utilities
│   ├── state.py            # state-vector and DMD-window utilities
│   └── weighting.py        # optional TKE-based spatial weighting
├── scripts/
│   ├── run_cylinder2d_dns.py
│   ├── run_cylinder2d_gdmd.py
│   ├── run_pinball_dns.py
│   ├── run_pinball_gdmd.py
│   ├── run_ptv_mean.py
│   └── run_ptv_gdmd.py
├── examples/
│   ├── reproduce_cylinder2d.md
│   ├── reproduce_pinball.md
│   └── reproduce_ptv.md
└── tests/
    └── test_dmd.py
```

## Installation

The gDMD post-processing scripts and the experimental PTV workflow only require the 
usual Python scientific stack (`numpy`, `scipy`, `scikit-learn`, `matplotlib`, `ray`, 
and `tqdm`). The two DNS generation scripts additionally require a working 
FEniCSx/DOLFINx installation with PETSc, MPI, and Gmsh.

For the DNS examples, 
**first activate the environment in which FEniCSx/DOLFINx is installed**, 
then install this repository into that same environment. The recommended workflow
is therefore:

```bash
# Replace <fenicsx-env> by the name of your existing FEniCSx/DOLFINx environment.
mamba activate <fenicsx-env>

# From the root of this repository, i.e. the directory containing pyproject.toml:
python -m pip install -e .
```

You can check that the active environment is the correct one with:

```bash
python -c "import dolfinx, gmsh, petsc4py, mpi4py; print('DOLFINx environment OK')"
python -c "import gdmd; print(gdmd.__file__)"
```

If you do not already have a FEniCSx/DOLFINx environment, install one following
the official [DOLFINx installation instructions](https://docs.fenicsproject.org/dolfinx/main/python/installation.html).
The DOLFINx project also maintains platform-specific recommendations in the
[DOLFINx README](https://github.com/FEniCS/dolfinx#installation). 

A conda/mamba environment file is provided for convenience:

```bash
mamba env create -f environment.yml
mamba activate gdmd-examples
python -m pip install -e .
```

If the `fenics-dolfinx` package in `environment.yml` is not available for your
platform, create a FEniCSx/DOLFINx environment using the official instructions
above, activate that environment, and then run `python -m pip install -e .` from
the repository root.

When using Spyder or Jupyter, make sure that the kernel is attached to the same
FEniCSx/DOLFINx environment. In Spyder, either launch Spyder from the activated
environment or select a kernel/interpreter whose `sys.executable` belongs to that
environment. A quick check inside Spyder is:

```python
import sys
print(sys.executable)

import dolfinx
import gdmd
print(gdmd.__file__)
```

Do not install the legacy `fenics` package for these scripts; the DNS solvers use
the newer `dolfinx`/FEniCSx interface. The experimental PTV scripts do not require
DOLFINx, but they can be run from the same environment for convenience.

## Confined cylinder

Generate the DNS/particle-pair database:

```bash
mpirun -n 4 python scripts/run_cylinder2d_dns.py \
    --final-time 300 \
    --stdHdiv 3 \
    --newseed-num 8 \
    --output-dir data
```

Run gDMD:

```bash
python scripts/run_cylinder2d_gdmd.py \
    --input data/final_np_rmin03_T300_stdH03_nsnum08.npz \
    --output results/cylinder2d_gdmd_results.npz \
    --n-modes 10 \
    --n-neighbors 15 \
    --gap-weights inv_r \
    --xlim 0.3 1.5 \
    --ylim 0.05 0.35 \
    --nproc 8 \
    --save-figures results/figures_cylinder
```

## Fluidic pinball

Generate the DNS/particle-pair database:

```bash
mpirun -n 4 python scripts/run_pinball_dns.py \
    --final-time 1500 \
    --stdHdiv 6 \
    --newseed-num 8 \
    --output-dir data
```

Run gDMD:

```bash
python scripts/run_pinball_gdmd.py \
    --input data/final_np_pinball_rmin03_T1500_stdH06_nsnum08.npz \
    --output results/pinball_gdmd_results.npz \
    --n-modes 10 \
    --n-neighbors 25 \
    --gap-weights uniform \
    --xlim 1 10 \
    --ylim -3 3 \
    --nproc 8 \
    --save-figures results/figures_pinball
```

## Experimental rectangular-jet PTV case

Download or clone the experimental dataset from:

```text
https://github.com/DingShizhe/PTV-Dataset
```

The scripts expect the denoised particle-position files to be available as:

```text
field_data_denoised/0000.npy
field_data_denoised/0001.npy
...
```

Build the mean-flow grid in two passes:

```bash
python scripts/run_ptv_mean.py \
    --field-root field_data_denoised \
    --grid linear \
    --output data/meanvel_linspace.npz \
    --borders data/borders.npz

python scripts/run_ptv_mean.py \
    --field-root field_data_denoised \
    --grid custom \
    --reference-mean data/meanvel_linspace.npz \
    --output data/meanvel_customspace.npz \
    --borders data/borders.npz
```

Run the experimental exact-DMD/gDMD comparison:

```bash
python scripts/run_ptv_gdmd.py \
    --field-root field_data_denoised \
    --mean-file data/meanvel_customspace.npz \
    --output results/ptv_exact_vs_gdmd_results.npz \
    --n-modes 8 \
    --n-neighbors-gap 30 \
    --n-neighbors-vel 30 \
    --gap-weights inv_r \
    --xlim 100 450 \
    --y-start 50 \
    --use-tke-weighting \
    --nproc 8 \
    --save-figures results/figures_ptv
```

Use `--normalized-time` if frequencies should be reported in normalized sample
units rather than physical units based on the 10 kHz acquisition frequency.
