# Ligand Generalization (Leave-Ligands-Out) Experiment

This directory contains the hierarchical CGCNN implementation for assessing out-of-distribution (OOD) generalization to ligand groups absent from training. The source code is an English, comment-free transcription of the supplied ligand-holdout implementation. The underlying neural network, data processing, training strategy, and numerical defaults are unchanged.

## Included files

- `data_loader_triplet_angle.py`: crystal graph and feature construction from CIF files.
- `hierarchical_cgcnn_triplet_angle.py`: hierarchical CGCNN network definition and loss functions.
- `train.py`: pre-filtering, ligand-group holdout split, training, evaluation, and checkpoints.

## Dataset

Provide a dataset directory containing `data.csv` (or `training_data.csv`), `atom_init.json`, and a `cif_files/` folder. The model also supports optional `zval_map.json` and `ionization_energy_map.json` when the corresponding features are enabled.

The CSV must identify each CIF file using a supported CIF filename column. In `joint` mode, `adsorption_energy` is preferred when available, while legacy target aliases remain supported. **The code retains `overpotential` as an internal target key for backward compatibility, even when the CSV supplies `adsorption_energy`; no conversion between adsorption energy and overpotential is performed.**

## Run

From the directory containing the three Python files:

```bash
python train.py ../../data --task joint --epochs 100 --seed 42 --reprocess --save-dir ligand_ood_results
```

To hold out specific ligand names:

```bash
python train.py ../../data --task joint --test-ligands THQ BHT --reprocess --save-dir ligand_ood_results
```

Use only ligand labels actually detected from your CIF naming convention. The script derives each ligand label from the filename segments following the first hyphen. Confirm that this naming rule matches the dataset before interpreting the split as ligand generalization.

## Split details

The script first removes samples without valid supervised targets. It then holds out three randomly selected ligand groups (or groups specified with `--test-ligands`) for OOD testing. The remaining ligand groups are divided into training and validation sets, with **15% of the remaining samples assigned to validation**. Thus, the final global proportions are **not guaranteed to be 70:15:15**.

## Original defaults

- Seed: `42`
- Early-stopping patience: `20` epochs
- Maximum epochs: `100`
- Batch size: `32`
- Learning rate: `1e-3`
- Graph cutoff: `8.0` Å
- Maximum neighbors: `12`
- Coordination number (`--use-cn`): disabled unless explicitly passed in this uploaded variant

The code's internal identifiers are intentionally preserved to avoid changing behavior. These source files have been syntax-checked, but training on the original CIF dataset and end-to-end numerical reproducibility have not been verified here.
