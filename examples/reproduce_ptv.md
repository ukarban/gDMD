# Reproduce the experimental rectangular-jet PTV example

Download or clone the public dataset from:

```text
https://github.com/DingShizhe/PTV-Dataset
```

The scripts expect the denoised particle-position files to be available locally
as:

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

Run exact DMD and gDMD:

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

Optional flags:

- `--normalized-time`: use unit sampling time instead of the physical 10 kHz
  acquisition interval.
- `--symmetry-augmentation`: append a mirrored copy of the particle fields about
  the estimated jet centreline.
- `--nproc 1`: run serially without Ray.
