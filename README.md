# 2D-MOF-GNN-H2O2

**Graph neural network models and density functional theory (DFT) data for *OOH adsorption-energy prediction in two-dimensional metal–organic frameworks (2D MOFs).**

This repository accompanies the research project **“Mechanistic Study on the Selective Generation of H2O2 in Metal-Organic Framework Electrocatalysis via Hierarchical Graph Neural Networks.”** The study investigates candidate catalysts for hydrogen peroxide (H2O2) production through the two-electron oxygen reduction reaction (2e− ORR).

The dataset comprises **270 candidate 2D MOF structures** spanning 27 metal centers and 10 local coordination environments. This repository provides three CGCNN-based pipelines: (i) a structure-only baseline, (ii) a hierarchical structure-plus-electronic-properties model, and (iii) an out-of-distribution ligand-holdout evaluation of hierarchical CGCNN. See [Reproducibility notes](#reproducibility-notes) for the scope and limitations of the released materials.

## Repository layout

```text
2D-MOF-GNN-H2O2/
├── data/
│   ├── cif_files/                         # 270 crystal structures in CIF format
│   ├── data.csv                           # DFT-derived properties and labels
│   ├── atom_init.json                     # atomic feature initialization
│   ├── zval_map.json                      # valence-electron feature lookup
│   └── ionization_energy_map.json         # ionization-energy feature lookup
├── models/
│   ├── structure_only/
│   │   ├── data_loader_triplet_angle.py
│   │   ├── hierarchical_cgcnn_triplet_angle.py
│   │   ├── train.py
│   │   └── README.md
│   ├── structure_electronic/
│   │   ├── data_loader_triplet_angle.py
│   │   ├── hierarchical_cgcnn_triplet_angle.py
│   │   ├── train.py
│   │   └── README.md
│   └── ligand_generalization/
│       ├── data_loader_triplet_angle.py
│       ├── hierarchical_cgcnn_triplet_angle.py
│       ├── train.py
│       └── README.md
├── .gitignore
├── README.md
├── requirements.txt
├── environment.yml
├── environment-record.yml
└── ENVIRONMENT_NOTES.md
```

The directory tree describes the intended GitHub layout. Generated `processed/` graph caches, training logs, model checkpoints, and plots are not required to be tracked in Git unless explicitly provided for reproducibility.

## Data and prediction targets

Each row of `data/data.csv` corresponds to a material and must resolve to an existing `.cif` file in `data/cif_files/`. The **recommended English column names** are:

| Column | Meaning | Used by |
| --- | --- | --- |
| `cif_filename` | Name of the CIF file (with or without `.cif`) | All three models |
| `adsorption_energy` | DFT-calculated *OOH adsorption energy (eV); regression target | All three models |
| `work_function` | Work function (eV) | Hierarchical and ligand-holdout models |
| `d_band_center` | d-band center (eV) | Hierarchical and ligand-holdout models |

The standard regression target column is `adsorption_energy`. Some data loaders also accept the equivalent Chinese column name `吸附能`. For the hierarchical pipeline, the `electronic` task requires the two electronic-property columns above. The `joint` task uses the *OOH adsorption-energy label and, where available, electronic-property labels.

**Prediction target:** `adsorption_energy` is the DFT-calculated *OOH adsorption energy, expressed in eV. All three models use this quantity as their supervised regression target. Keep the adsorption-energy definition and units consistent throughout the dataset, training, and evaluation.

## Computational environment

The computational environment used in this study included **Python 3.10.19**, **PyTorch 2.9.0+cu128**, **PyTorch Geometric 2.5.2**, **NumPy 2.2.6**, **scikit-learn 1.7.2**, and **pymatgen 2025.10.7**.

All model training and evaluation experiments were performed on **CPUs**.

The software dependencies and environment configurations are provided in `requirements.txt`, `environment.yml`, and `environment-record.yml` for reference.

## Model 1: Structure-only CGCNN baseline

Implemented in [`models/structure_only/`](models/structure_only/). This baseline learns from crystal-graph and atomic attributes, including coordination number, atomic valence-electron and ionization-energy lookup features. It does **not** take DFT-calculated work function or d-band center as an additional supervised electronic-anchor branch.

From the **repository root**, run:

```bash
python models/structure_only/train.py data \
  --epochs 300 \
  --patience 30 \
  --seed 42 \
  --reprocess \
  --save-dir checkpoints/structure_only
```

Defaults in this implementation include batch size 16, a maximum of 300 epochs, learning rate 1e-3, cutoff radius 8 Å, up to 12 neighbors, an intended 70/15/15 train/validation/test split, and early-stopping patience 30.

The program writes `split_assignments.csv`, the best model weights (`best_direct_model.pt`), `train_predictions.csv`, `test_predictions.csv`, parity-plot data, and plots to the chosen output directory.

To compare against an existing split, use **both** `--train-csv` and `--test-csv` with prediction files containing sample identifiers:

```bash
python models/structure_only/train.py data \
  --epochs 300 --patience 30 --seed 42 --reprocess \
  --train-csv checkpoints/structure_electronic/train_predictions.csv \
  --test-csv checkpoints/structure_electronic/test_predictions.csv \
  --save-dir checkpoints/structure_only_fixed_split
```

In this mode, samples absent from both supplied lists become the validation set; verify the split sizes and the generated assignment file before comparing metrics.

## Model 2: Hierarchical structure + electronic properties CGCNN

Implemented in [`models/structure_electronic/`](models/structure_electronic/). The first stage predicts **work function** and **d-band center** from crystal representations. Joint training combines structural representations and predicted electronic features to regress the adsorption-energy label. The code's current Stage 1 has **two** electronic-property targets (not a third Fermi-level output).

**Stage 1 — electronic properties:**

```bash
python models/structure_electronic/train.py data \
  --task electronic \
  --epochs 300 --patience 30 --seed 42 --reprocess \
  --save-dir checkpoints/electronic_stage1
```

**Stage 2 — joint adsorption-energy regression:**

```bash
python models/structure_electronic/train.py data \
  --task joint \
  --init-electronic-ckpt checkpoints/electronic_stage1/best_model.pt \
  --epochs 300 --patience 30 --seed 42 --reprocess \
  --save-dir checkpoints/structure_electronic
```

This implementation defaults to batch size 16, maximum 300 epochs, learning rate 1e-3, cutoff radius 8 Å, up to 12 neighbors, Gaussian radial distance features (64 bins), CN enabled, an intended 70/15/15 train/validation/test split, and early-stopping patience 30.

The hierarchical script records configuration (`args.json`), label normalization statistics, model checkpoints, prediction tables, training history, test metrics, and plots under `--save-dir`.

## Model 3: Ligand generalization (leave-ligands-out evaluation)

Implemented in [`models/ligand_generalization/`](models/ligand_generalization/). This script evaluates whether a hierarchical CGCNN can predict the adsorption-energy target for **ligand groups absent from its training set**. It uses the same types of CIF-derived crystal graphs and electronic-property labels as the hierarchical pipeline but adopts a separate **out-of-distribution (OOD) splitting procedure**. It is a distinct experiment, not a conventional random 70/15/15 benchmark.

**Default ligand-holdout run (from the repository root):**

```bash
python models/ligand_generalization/train.py data \
  --task joint \
  --epochs 300 --patience 30 --seed 42 --reprocess \
  --save-dir checkpoints/ligand_generalization
```

The script filters samples with invalid supervised targets, identifies ligand groups from CIF filenames, and chooses **three ligand groups** for an unseen test set using the specified seed. It then sets aside **15% of samples from the remaining ligand groups** for validation; the rest form the training set. The overall train/validation/test percentages therefore depend on the relative sizes of the held-out ligand groups and are **not fixed at 70/15/15**. The ligand split should be documented alongside any reported OOD metrics.

**Hold out specific ligand groups:**

```bash
python models/ligand_generalization/train.py data \
  --task joint --test-ligands LIGAND_A LIGAND_B LIGAND_C \
  --epochs 300 --patience 30 --seed 42 --reprocess \
  --save-dir checkpoints/ligand_generalization_specified
```

Replace the illustrative `LIGAND_A`/`LIGAND_B`/`LIGAND_C` arguments with **exact ligand labels reported by the program**. For each CIF filename, the current code removes the `.cif` suffix, splits the basename at hyphens, and uses the joined segments after the first hyphen as the ligand label; for example, `Metal-Ligand-A.cif` maps to `Ligand-A`. Inspect the printed detected groups and check this parsing rule against the actual CIF naming convention before interpreting the experiment.

The ligand-holdout implementation defaults to batch size **16**, learning rate **1e-3**, cutoff radius **8 Å**, up to **12 neighbors**, seed **42**, maximum **300 epochs**, and early-stopping patience **30**. The three pipelines may use different preprocessing and dataset-splitting configurations.

**Prediction target:** the ligand-holdout model predicts DFT-calculated *OOH adsorption energy (eV). The target field is named `adsorption_energy` in the model, loss calculations, evaluation metrics, and output files. The CSV reader accepts `adsorption_energy` or the equivalent Chinese header `吸附能`.

For details specific to the ligand-holdout script, see [`models/ligand_generalization/README.md`](models/ligand_generalization/README.md).

## Reproducibility notes

- **Run commands from the repository root** to ensure that `data` resolves correctly. All three model directories contain their own `train.py` and supporting modules.
- **Check that every filename in `data.csv` matches its CIF file** and that the target and electronic labels have the units and definitions intended by the study.
- **Graph cache warning:** all three pipelines can use `data/processed/`. The graph caches are not model-configuration-aware. Use `--reprocess` when switching pipelines, changing graph features, changing sample order, or changing labels. This flag deletes existing processed graph cache files. Do not run multiple pipelines against the same cache at the same time.
- **Splits:** matching random seeds alone does not guarantee identical sample membership across pipelines, particularly if validity filtering differs. Compare sample-ID assignments before reporting head-to-head metrics. The structure-only script supports externally fixed train/test membership; the standard hierarchical training script uses its own single random split; ligand generalization holds out entire ligand groups and is not directly comparable to an ordinary random-split test set.
- **Environment and reproducibility:** small numerical differences can occur across package versions, devices, and hardware. Report the exact versions and split assignments used for published results.

## Related manuscript and citation

**Related manuscript:**

Yan, M.; Du, Y.; Li, Y.; Ge, X.; Zhou, W.; Zhang, R.; Hu, J.; Hao, G. **Mechanistic Study on the Selective Generation of H2O2 in Metal-Organic Framework Electrocatalysis via Hierarchical Graph Neural Networks.** *Manuscript draft; final journal citation and DOI are not yet available.*

If you use the code or data from this repository, please acknowledge the GitHub repository and cite the associated research article when its publication details become available. The final journal citation and DOI will be added to this section after publication.

Repository: https://github.com/33248/2D-MOF-GNN-H2O2

## Funding and license

This research was supported by the National Natural Science Foundation of China (Grant No. **22303031**).

**License status:** no software/data reuse license is specified by these documentation files. A public GitHub repository is not, by itself, a software or dataset license. Please add a `LICENSE` and, if appropriate, explicit dataset reuse terms after confirming the authors' preferred permissions.
