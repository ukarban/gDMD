# gDMD
Gap-based Dynamic Mode Decomposition package for DNS and PTV data analysis. The numerical and experimental datasets analyzed here consist of:

- **Direct Numerical Simulations (DNS)** of flow past a 2D cylinder using [FEniCSx](https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html)
- **Particle Tracking Velocimetry (PTV)** data of a rectangular jet from (https://github.com/DingShizhe/PTV-Dataset)

The codes allow reproduction of the results and figures in the [paper](http://dx.doi.org/10.13140/RG.2.2.34468.18561).

---

## 📁 Repository Structure

```
.
├── DNS/
│   ├── cylinder2d_gdmd.py   # Performs DNS using FEniCSx and applies gDMD
│   ├── dmd_dict.py          # Contains core DMD functions
│   ├── perform_gdmd.py      # Standalone script for performing gDMD on existing DNS data
├── PTV/
│   ├── ptv_calc_meanvel.py  # Processes PTV data, identifies jet center/lip, and grids the field
│   └── ptv_perform_gdmd.py  # Performs gDMD on the PTV dataset
```

---

## 🚀 How to Run

### Dependencies

- Python 3.10+
- [FEniCSx](https://docs.fenicsproject.org/)
- NumPy
- SciPy
- h5py
- mpi4py (used only in `cylinder2d_gdmd.py`)
- matplotlib (optional, for plotting)
- Sci-kit learn
- **Ray** (for parallel gDMD)

You may install python packages using:

```bash
pip install -r requirements.txt
```

> **Note:** FEniCSx installation requires a working Docker/Singularity or direct installation from [FEniCSx documentation](https://docs.fenicsproject.org/).

---

### Example Workflow

#### 🔷 1. Run DNS and gDMD:
```bash
mpirun -np 4 python3 cylinder2d_gdmd.py
```

This will simulate 2D flow past a cylinder, save the velocity snapshots, and perform gDMD on the result.

#### 🔷 2. Perform gDMD on precomputed DNS data:
```bash
python3 perform_gdmd.py
```

#### 🔷 3. Process PTV Data and Compute Mean Flow:
```bash
python3 ptv_calc_meanvel.py
```

#### 🔷 4. Perform gDMD on PTV Data:
```bash
python3 ptv_perform_gdmd.py
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🔗 Citation

If you use this code, please cite the paper:

> *U. Karban, Dynamic mode decomposition using standard particle image velocimetry data, preprint, 2025. DOI: 10.13140/RG.2.2.34468.18561*

---

## 📬 Contact

For questions, please contact:  
**Ugur Karban**  
Email: [ukarban@metu.edu.tr]
