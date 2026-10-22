# Hierarchical CGCNN for *OOH Adsorption-Energy Prediction

This directory contains the hierarchical crystal graph convolutional neural network (CGCNN) for two-stage electronic-property prediction and *OOH adsorption-energy regression in two-dimensional metal–organic frameworks.

## Dataset

Use the repository-level `data/` directory containing `data.csv`, `cif_files/`, `atom_init.json`, `zval_map.json`, and `ionization_energy_map.json`.

The recommended CSV columns are `cif_filename`, `adsorption_energy`, `work_function`, and `d_band_center`. The `adsorption_energy` target is the DFT-calculated *OOH adsorption energy in eV. The electronic-property targets are work function and d-band center, also in eV.

## Run from the repository root

**Stage 1 — electronic-property prediction**

```bash
python models/structure_electronic/train.py data \
  --task electronic \
  --epochs 300 --patience 30 --seed 42 --reprocess \
  --save-dir checkpoints/electronic_stage1
```

**Stage 2 — joint adsorption-energy prediction**

```bash
python models/structure_electronic/train.py data \
  --task joint \
  --init-electronic-ckpt checkpoints/electronic_stage1/best_model.pt \
  --epochs 300 --patience 30 --seed 42 --reprocess \
  --save-dir checkpoints/structure_electronic
```

The joint stage initializes the electronic-property predictor from the Stage 1 checkpoint.

## Default settings

- Maximum epochs: `300`
- Early-stopping patience: `30`
- Batch size: `16`
- Learning rate: `1e-3`
- Random seed: `42`
- Cutoff radius: `8` Å
- Maximum neighbors: `12`
- Gaussian radial distance features: `64` bins
- Coordination-number (CN), ZVAL, and ionization-energy node features: enabled by default
- Intended training/validation/test split: `70/15/15`

ZVAL and ionization-energy lookup tables are read from `data/zval_map.json` and `data/ionization_energy_map.json` by default. To disable either auxiliary node feature, use `--no-zval` or `--no-ionization-energy`.

The learning-rate scheduler has a separate patience setting of 10 epochs. Graph features and cached files should be rebuilt with `--reprocess` when changing model feature settings.

## Outputs

Training outputs are stored under `--save-dir` and include the argument configuration (`args.json`), normalization statistics, model checkpoints, prediction tables, training history, evaluation metrics, and plots.
