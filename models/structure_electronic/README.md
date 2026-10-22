# 2D-MOF-GNN-H2O2 — Adsorption Energy Target

This version regresses an adsorption-energy label rather than overpotential. All Python source messages and CLI help text are in English; original model and training operations have not been changed. It preserves the original two-stage Hierarchical CGCNN architecture: electronic-property prediction (`work_function`, `d_band_center`) and joint adsorption-energy prediction.

## Data layout

```
data/
├── training_data.csv
├── atom_init.json
└── cif_files/
    ├── example_1.cif
    └── ...
```

Required CSV fields for `--task joint`: `cif_filename`, `adsorption_energy`, `work_function`, `d_band_center` (use actual filenames; the electronic fields are used by the joint electronic-property losses). The target column also accepts `adsorption_free_energy`, `ΔG*OOH`, `ΔG_OOH`, or `dG_OOH`. The target value is used directly, without transformation; only provide actual adsorption-energy or adsorption-free-energy labels in eV, according to your experiment's definition.

## Commands

```
python train.py data --task electronic --reprocess --save-dir ckpt_electronic
python train.py data --task joint --reprocess --init-electronic-ckpt ckpt_electronic/best_model.pt --save-dir ckpt_adsorption
```

Early stopping patience defaults to **30 epochs** without improvement in validation loss (`--patience 30`). The learning-rate scheduler retains its original, separate patience of 10 epochs.

Coordination-number (`CN`) node features are enabled by default; no `--use-cn` argument is needed. The current model uses coordination-number statistics in its global feature extractor, so this package keeps CN enabled. Use `--reprocess` to rebuild old graph caches.

Install required dependencies (PyTorch, PyTorch Geometric, pymatgen, pandas, scikit-learn, matplotlib, numpy) in an appropriate environment before running. The `--reprocess` flag removes stale cached graph objects from older overpotential-labeled versions. Check data and physical quantities prior to training. The repository's original triplet-angle implementation and training pipeline have not been independently verified in a complete training run.

Outputs from joint training use `adsorption_energy`, including prediction CSV columns, evaluation metrics, plots and JSON results.

## Notes on this edition

- Source: the previously supplied adsorption-energy **structure + electronic properties** HierarchicalCGCNN version with coordination-number features enabled and default early-stopping patience 30.
- The three Python source files contain no comments or docstrings. All Chinese display text is translated to English. Legacy Chinese CSV aliases remain supported internally by Unicode escape strings so that their values are unchanged at runtime.
- Stage 1 predicts `work_function` and `d_band_center`. Joint training predicts the adsorption-energy target and electronic properties as defined in the original model.
- For local data in `training_dataset/1`, replace `data` in the example commands with `training_dataset/1`.
- Exact training results still depend on data, caches, splits and environment. Full training with the 270 CIF files has not been executed for this edition.
