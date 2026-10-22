# Direct CGCNN for Adsorption Energy Prediction in 2D MOFs

This folder is a **structure-only** model for adsorption-energy regression, not the full two-stage electronic-property/adsorption-energy model.

## Data

Place `data.csv`, `atom_init.json`, `zval_map.json`, `ionization_energy_map.json`, and `cif_files/` under a single data directory (e.g. `training_dataset/1/`).

The CSV requires a CIF filename column (`cif_filename`, `cif_file`, or a column whose name contains `cif` or `file`) and one of the labels: `adsorption_energy`, `adsorption_free_energy`, `ΔG*OOH`, `ΔG_OOH`, `delta_g_ooh`. The model learns the numeric labels **as-is**; it does not convert adsorption energies into overpotentials. Ensure the selected label represents the quantity described in your paper (e.g. ΔG*OOH if using adsorption free energies).

## Train

```
python train.py training_dataset/1 --epochs 200 --patience 30 --reprocess --save-dir checkpoints/direct_adsorption
```

Default: 70/15/15 random split, seed 42, batch size 16, cutoff 8 Å, max neighbors 12, coordination features enabled, ZVAL/ionization features enabled, triplet-angle disabled, early-stopping patience 30.

To reproduce the same train/test partition from another model, supply both CSV files:

```
python train.py training_dataset/1 --epochs 200 --reprocess --train-csv PATH_TO_TRAIN_PREDICTIONS.csv --test-csv PATH_TO_TEST_PREDICTIONS.csv --save-dir checkpoints/direct_adsorption_fixed
```

In the fixed-split mode, sample names listed in those two CSV files determine train and test sets; the remaining valid samples form the validation set. Their counts are data-dependent. The program writes `split_assignments.csv` to the selected output folder.

The supplied `hierarchical_cgcnn_triplet_angle.py` retains **legacy hierarchical overpotential/SISSO classes** from the input model source, but `train.py` imports and trains only `DirectAdsorptionEnergyPredictor`. Do not use the legacy hierarchical components to claim adsorption-energy reproduction without separately validating/adapting them.

The scripts were statically checked for Python syntax and import/class-name consistency. A real 270-CIF training run has not been performed here; PyTorch Geometric/pymatgen are required. Use `--reprocess` when changing labels or graph featurization to discard stale `processed/data_*.pt` cache items.

## Localization

The Python scripts use English status messages and error text. Compatibility with legacy Chinese CSV header aliases is retained internally; the recommended CSV headers are `cif_filename` and `adsorption_energy`. English-language changes do not affect model parameters, sample selection, or training computations.
