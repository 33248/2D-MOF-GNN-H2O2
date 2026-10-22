# Structure-Only CGCNN for *OOH Adsorption-Energy Prediction

This directory contains the structure-only CGCNN baseline for predicting DFT-calculated *OOH adsorption energy (eV) in two-dimensional metal–organic frameworks.

## Dataset

Use the repository-level `data/` directory containing `data.csv`, `atom_init.json`, `cif_files/`, `zval_map.json`, and `ionization_energy_map.json`. The recommended dataset columns are `cif_filename` and `adsorption_energy`. The supervised regression target is *OOH adsorption energy in eV.

## Run from the repository root

```bash
python models/structure_only/train.py data \
  --epochs 300 --patience 30 --seed 42 --reprocess \
  --save-dir checkpoints/structure_only
```

To use train/test membership exported by the structure-plus-electronic-properties pipeline:

```bash
python models/structure_only/train.py data \
  --epochs 300 --patience 30 --seed 42 --reprocess \
  --train-csv checkpoints/structure_electronic/train_predictions.csv \
  --test-csv checkpoints/structure_electronic/test_predictions.csv \
  --save-dir checkpoints/structure_only_fixed_split
```

The sample IDs provided by the two CSV files determine the training and test sets. Remaining eligible samples form the validation set. Review `split_assignments.csv` before comparing results across models.

## Default settings

- Maximum epochs: `300`
- Early-stopping patience: `30`
- Batch size: `16`
- Learning rate: `1e-3`
- Random seed: `42`
- Cutoff radius: `8` Å
- Maximum neighbors: `12`
- Intended training/validation/test split: `70/15/15`
- Structural and atomic node features include coordination number, ZVAL, and ionization energy

The structure-only model does not use the DFT-calculated work function or d-band center as supervised electronic-property prediction targets.

## Outputs and cache

The selected output directory contains `split_assignments.csv`, `best_direct_model.pt`, training/test prediction tables, and plotting data. Use `--reprocess` after changing graph features, labels, or data order to rebuild cached graphs in `data/processed/`.
