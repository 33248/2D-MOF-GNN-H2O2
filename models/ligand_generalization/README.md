# Ligand Generalization (Leave-Ligands-Out) — Adsorption-Energy Prediction

This directory contains the hierarchical crystal graph convolutional neural network (CGCNN) implementation for out-of-distribution ligand-holdout evaluation of *OOH adsorption-energy regression.

The model predicts the numeric adsorption-energy target supplied by the dataset. It does not derive other catalytic activity metrics from the input labels; those must be calculated separately using the convention in the associated study.

## Files

- `train.py`: dataset reading, filtering, ligand-group holdout, training, testing, and prediction export.
- `data_loader_triplet_angle.py`: conversion of CIF structures into crystal graphs.
- `hierarchical_cgcnn_triplet_angle.py`: hierarchical model, electronic-property heads, adsorption-energy predictor, and loss functions.

## Dataset

Use the repository-level `data/` folder containing `data.csv`, `atom_init.json`, and `cif_files/` with its CIF structures. Optional element-property files are `zval_map.json` and `ionization_energy_map.json`.

The supervised target column is `adsorption_energy` (eV). The input CSV reader also accepts the equivalent Chinese header `吸附能`. The target is the DFT-calculated *OOH adsorption energy used in this study. Internal training variables, plots, model outputs, and exported metric names use `adsorption_energy`.

## Run (from the repository root)

```bash
python models/ligand_generalization/train.py data \
  --task joint --epochs 300 --patience 30 --seed 42 --reprocess \
  --save-dir checkpoints/ligand_generalization
```

To choose held-out ligand groups explicitly:

```bash
python models/ligand_generalization/train.py data \
  --task joint --test-ligands LIGAND_A LIGAND_B LIGAND_C \
  --epochs 300 --patience 30 --seed 42 --reprocess \
  --save-dir checkpoints/ligand_generalization_custom
```

Replace ligand placeholders with exactly the groups recognized from your CIF filenames. This script takes the filename portion after the first `-` as the ligand group label; confirm that this matches the naming in your dataset.

## Data splitting

The program filters out invalid target labels before splitting. With no explicit `--test-ligands`, it selects three ligand groups for OOD testing using the supplied seed. It then reserves 15% of the remaining samples for validation and uses the rest for training. Consequently, the overall dataset does **not** have a fixed 70:15:15 division.

## Default settings

- Random seed: `42`
- Maximum epochs: `300`
- Early-stopping patience: `30`
- Batch size: `16`
- Learning rate: `1e-3`
- Cutoff: `8` Å
- Maximum neighbors: `12`
- Coordination-number (CN) feature: enabled by default (`--no-cn` disables it)
- Electronic targets: `work_function` and `d_band_center`
- Adsorption-target loss weight option: `--lambda-adsorption-energy`
