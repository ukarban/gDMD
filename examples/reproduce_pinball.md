# Reproduce the fluidic-pinball gDMD example

Generate the DNS/particle-pair database:

```bash
mpirun -n 4 python scripts/run_pinball_dns.py \
    --final-time 1500 \
    --stdHdiv 6 \
    --newseed-num 8 \
    --output-dir data
```

Run the baseline gDMD analysis:

```bash
python scripts/run_pinball_gdmd.py \
    --input data/final_np_pinball_rmin03_T1500_stdH06_nsnum08.npz \
    --output results/pinball_gdmd_results.npz \
    --n-modes 10 \
    --n-neighbors 25 \
    --xlim 1 10 \
    --ylim -3 3 \
    --gap-weights uniform \
    --nproc 8 \
    --save-figures results/figures_pinball
```

Run the optional TKE-weighted gDMD analysis:

```bash
python scripts/run_pinball_gdmd.py \
    --input data/final_np_pinball_rmin03_T015_stdH06_nsnum08.npz \
    --output results/pinball_gdmd_results_tke.npz \
    --use-tke-weighting \
    --nproc 8 \
    --save-figures results/figures_pinball_tke
```

For a fast smoke test, reduce `--final-time`, but such a run will not reproduce
the reported eigenspectra or mode shapes.
