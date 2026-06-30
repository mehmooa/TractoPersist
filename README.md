# TractoPersist

**Persistent Homological Characterisation of SIFT2-Filtered Structural Connectomes Across the Alzheimer's Disease Spectrum**

> Submitted to CDMRI'26 Workshop, MICCAI 2026, Strasbourg

---

## Overview

TractoPersist is a novel framework combining:
- **SIFT2-filtered structural connectomes** from DTI tractography
- **Persistent homology** (H0 + H1 topological features)
- **Hybrid GAT + Graph Transformer** neural network
- **ROI-specific FA/MD/RD** microstructure metrics

Applied to **ADNI-3** (n=214: 25 AD, 89 MCI, 100 CN, 54-gradient protocol).

---

## Key Results

| Task | AUC | Accuracy | vs Best Baseline |
|------|-----|----------|-------------------|
| AD vs CN | **0.80** | 82.0% | +0.16 AUC |
| MCI vs AD | **0.70** | 80.0% | +0.13 AUC |
| CN vs MCI | **0.63** | 65.1% | +0.01 AUC |

- MD: p=0.0004*** (AD vs CN)
- RD: p=0.0005*** (AD vs CN)
- H1 topology correlates with MD variability (r=0.20, p=0.003)

---

## Repository Structure

```
TractoPersist/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── figures/                                # Final figures used in the paper
│   ├── fig3_statistics.png                 # Violin plots / group statistics
│   ├── fig4_correlation.png                # Topology–DTI correlation heatmap
│   ├── fig5_pipeline.png                   # Pipeline diagram (simple version)
│   ├── fig6_persistence.png                # Persistence diagrams
│   └── fig_pipeline_professional.png       # Publication-quality pipeline figure
├── src/
│   ├── preprocessing/
│   │   ├── step0_brain_masking.sh          # Standalone brain mask QC step
│   │   ├── pipeline.sh                     # Denoise→Gibbs→mask→CSD→tractography→SIFT2→connectome
│   │   ├── extract_metrics.sh              # FA/MD/RD tensor maps
│   │   ├── step0b_atlas_registration.sh    # Harvard-Oxford atlas → native DWI space (FSL flirt)
│   │   ├── step1_connectome_features.py    # Graph-theoretic features from connectome matrices
│   │   ├── step2_persistent_homology.py    # H0/H1 topological features (Ripser)
│   │   └── step3_combine_features.py       # Merge connectome+homology+DTI → combined_features.csv
│   ├── features/
│   │   ├── step_topology_analysis.py       # Persistent homology statistics
│   │   ├── step_all_binary.py              # Baseline ML classifiers
│   │   └── step_advanced_binary.py         # Extended classification suite
│   ├── models/
│   │   ├── tractopersist_model.py          # Core GAT+GT architecture
│   │   ├── tractopersist_v2.py             # Full training pipeline
│   │   └── step_ablation.py                # Ablation study (component contribution)
│   └── figures/
│       ├── step_fig1_brain_maps.py         # DTI microstructure maps
│       ├── step_fig2_connectome.py         # Connectome matrix + glass brain
│       ├── step_fig3_statistics.py         # Violin plots / group stats
│       ├── step_fig4_correlation.py        # Topology–DTI correlation heatmap
│       ├── step_fig5_pipeline.py           # Pipeline diagram
│       ├── step_fig6_persistence.py        # Persistence diagrams
│       ├── step_pipeline_with_images.py    # Publication pipeline figure
│       └── step_run_all_figures.py         # Run all figure scripts
└── results/                                # Output CSVs (not tracked — see .gitignore)
```

---

## Requirements

### System
- Ubuntu 22.04 (WSL2 or native)
- Python 3.9+
- CUDA GPU recommended

### Software
- MRtrix3 3.0.8
- dcm2niix
- Harvard-Oxford Atlas (FSL)

### Python Packages
```bash
pip install -r requirements.txt
```

---

## Pipeline

### Step 0 — Brain Masking (QC, optional standalone)
```bash
bash src/preprocessing/step0_brain_masking.sh
```
Extracts mean b0 + brain mask per subject for visual QC before committing to the full pipeline.

### Step 1 — DTI Preprocessing → SIFT2 Connectome (Ubuntu/WSL)
```bash
bash src/preprocessing/pipeline.sh
```
Runs per subject: `dwidenoise` → `mrdegibbs` → `dwi2mask` → `dwi2response tournier` → `dwi2fod csd` → `tckgen` (iFOD2, 100K) → `tcksift2` → `tck2connectome` (48-ROI)

### Step 2 — DTI Microstructure Maps
```bash
bash src/preprocessing/extract_metrics.sh
```
Extracts FA, MD, RD maps per subject via `dwi2tensor` + `tensor2metric`.

### Step 3 — Atlas Registration to Native Space
```bash
bash src/preprocessing/step0b_atlas_registration.sh
```
Registers the Harvard-Oxford 48-ROI atlas (MNI) into each subject's native DWI space (FSL `flirt`, nearest-neighbour), required for per-ROI FA/MD/RD extraction in Step 6.

### Step 4 — Connectome Graph Features
```bash
python src/preprocessing/step1_connectome_features.py
```
Node strength/degree and global graph metrics from each SIFT2-weighted connectome → `connectome_features.csv`.

### Step 5 — Persistent Homology Features
```bash
python src/preprocessing/step2_persistent_homology.py
```
Vietoris-Rips filtration (Ripser) on each connectome → 13 H0/H1 topological descriptors → `homology_features.csv`.

### Step 6 — Combine All Features
```bash
python src/preprocessing/step3_combine_features.py
```
Merges connectome, homology, and per-ROI FA/MD/RD features into one table → `combined_features.csv` (the input to everything below).

### Step 7 — Topology Analysis & Statistics
```bash
python src/features/step_topology_analysis.py
```

### Step 8 — Baseline Classification
```bash
python src/features/step_all_binary.py
python src/features/step_advanced_binary.py
```

### Step 9 — TractoPersist Model Training
```bash
python src/models/tractopersist_v2.py
```
Hybrid GAT+GT with topology attention, 5-fold CV, Youden-optimal thresholding.

### Step 10 — Ablation Study
```bash
python src/models/step_ablation.py
```
Compares GAT-only / GT-only / no-topology / full model variants.

### Step 11 — Figures
```bash
python src/figures/step_run_all_figures.py
```

---

## Data

Data from [ADNI](https://adni.loni.usc.edu/) (registered access required). Raw imaging data and per-subject connectomes are **not** included in this repository — only code and aggregated results.

**Protocol**: Axial DTI, 54 gradients, b=1000 s/mm², 2×2×2 mm³

**Cohort**: AD n=25, MCI n=89, CN n=100 (54-gradient baseline subset)

---

## Citation

```bibtex
@inproceedings{tractopersist2026,
  title={TractoPersist: Persistent Homological Characterisation
         of SIFT2-Filtered Structural Connectomes Across the
         Alzheimer's Disease Spectrum},
  booktitle={CDMRI Workshop, MICCAI 2026},
  year={2026}
}
```

## Key References
- Smith et al. SIFT2. *NeuroImage* 119:338-351, 2015
- Tournier et al. MRtrix3. *NeuroImage* 202:116137, 2019
- Bauer. Ripser. *J Appl Comput Topol* 5:391-423, 2021

## License
MIT — see [LICENSE](LICENSE)
