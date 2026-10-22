import argparse
import os
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random


from data_loader_triplet_angle import MOFDataset
from hierarchical_cgcnn_triplet_angle import DirectAdsorptionEnergyPredictor


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dataset_flexible(data_path):
    data_path = Path(data_path)


    csv_file = data_path / "training_data.csv"
    if not csv_file.exists():
        csv_file = data_path / "data.csv"
    if not csv_file.exists():
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"❌ No CSV file found in {data_path}")
        csv_file = csv_files[0]

    df = pd.read_csv(csv_file)
    adsorption_cols = ['adsorption_energy', 'adsorption_free_energy', '\u5438\u9644\u80fd', '\u5438\u9644\u81ea\u7531\u80fd', 'ΔG*OOH', 'ΔG_OOH', 'delta_g_ooh']
    energy_col = next((name for name in adsorption_cols if name in df.columns), None)
    if energy_col is None:
        raise ValueError(f"CSV is missing an adsorption-energy target column. Supported columns: {adsorption_cols}; available columns: {df.columns.tolist()}")
    cif_col = next((col for col in df.columns if 'cif' in col.lower() or 'file' in col.lower()), df.columns[0])


    cif_subfolder_exists = (data_path / "cif_files").exists()
    if cif_subfolder_exists:
        print("🔍 Found a cif_files subdirectory; resolving CIF paths...")

    cif_files = []
    properties_dict = {}

    for _, row in df.iterrows():
        cif_name = str(row[cif_col]).strip()
        if not cif_name.endswith('.cif'):
            cif_name += '.cif'


        if cif_subfolder_exists:
            dataset_cif_path = str(data_path / "cif_files" / cif_name)
        else:
            dataset_cif_path = str(data_path / cif_name)

        cif_files.append(dataset_cif_path)

        props = {'adsorption_energy': pd.to_numeric(row[energy_col], errors='coerce')}
        for auxiliary_key in ('fermi_level', 'work_function', 'd_band_center'):
            if auxiliary_key in df.columns:
                props[auxiliary_key] = row[auxiliary_key]


        properties_dict[dataset_cif_path] = props
        properties_dict[dataset_cif_path.replace('.cif', '')] = props
        properties_dict[cif_name] = props
        properties_dict[cif_name.replace('.cif', '')] = props

    return cif_files, properties_dict

def normalize_sample_name(x):
    x = str(x).strip()
    x = os.path.basename(x)
    if x.lower().endswith(".cif"):
        x = x[:-4]
    return x.lower()

def detect_name_col(df):
    candidates = [
        "sample_id", "Sample_Name", "sample_name", "sample",
        "name", "cif_file", "cif_filename", "filename", "cif\u6587\u4ef6\u540d"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = str(c).lower()
        if "sample" in cl or "cif" in cl or "file" in cl or "name" in cl:
            return c
    raise ValueError(f"No sample-name column found. Available columns: {df.columns.tolist()}")

def load_name_set(csv_path):
    df = pd.read_csv(csv_path)
    col = detect_name_col(df)
    return set(df[col].map(normalize_sample_name)), col

def make_split_from_prediction_csv(valid_data_list, train_csv, test_csv):
    train_names, train_col = load_name_set(train_csv)
    test_names, test_col = load_name_set(test_csv)

    overlap = train_names & test_names
    if overlap:
        raise ValueError(f"train_csv and test_csv contain overlapping samples. Examples: {sorted(overlap)[:20]}")

    sample_names = [getattr(d, "sample_name", f"sample_{i}") for i, d in enumerate(valid_data_list)]
    norm_names = [normalize_sample_name(x) for x in sample_names]

    train_idx = np.array([i for i, nm in enumerate(norm_names) if nm in train_names], dtype=int)
    test_idx = np.array([i for i, nm in enumerate(norm_names) if nm in test_names], dtype=int)
    used = set(train_idx.tolist()) | set(test_idx.tolist())
    val_idx = np.array([i for i in range(len(valid_data_list)) if i not in used], dtype=int)

    found_train = set(norm_names[i] for i in train_idx)
    found_test = set(norm_names[i] for i in test_idx)
    missing_train = sorted(train_names - found_train)
    missing_test = sorted(test_names - found_test)

    print("\n" + "=" * 40)
    print("🔒 Using external train/test CSV files for a fixed split")
    print(f"   train_csv: {train_csv}  sample column: {train_col}")
    print(f"   test_csv : {test_csv}  sample column: {test_col}")
    print(f"   Train matched: {len(train_idx)} / {len(train_names)}")
    print(f"   Test matched : {len(test_idx)} / {len(test_names)}")
    print(f"   Val inferred : {len(val_idx)}")
    if missing_train:
        print(f"   [WARN] {len(missing_train)} train_csv samples could not be matched; first 20: {missing_train[:20]}")
    if missing_test:
        print(f"   [WARN] {len(missing_test)} test_csv samples could not be matched; first 20: {missing_test[:20]}")
    print("=" * 40)

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("External CSV matching failed: Train or Test set is empty. Check the sample-name columns.")

    return train_idx, val_idx, test_idx, norm_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, default="training_dataset/1/")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/direct_single_70_15_15")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-csv", type=str, default=None, help="train_predictions.csv exported by another model, used to fix the structure-only training split")
    parser.add_argument("--test-csv", type=str, default=None, help="test_predictions.csv exported by another model, used to fix the structure-only test split")
    parser.add_argument("--reprocess", action="store_true")
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_save_dir = Path(args.save_dir)
    base_save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "⬛" * 30)
    print("🚀 Structure-only adsorption-energy prediction: single 70:15:15 split")
    print("⬛" * 30)


    set_seed(args.seed)

    cif_files, properties_dict = read_dataset_flexible(args.data_path)
    if args.reprocess:
        processed_dir = Path(args.data_path) / "processed"
        if processed_dir.exists():
            cleared = 0
            for cache_file in processed_dir.glob("data_*.pt"):
                cache_file.unlink()
                cleared += 1
            print(f"Removed {cleared} cached graph files")

    dataset = MOFDataset(
        root=args.data_path, cif_files=cif_files, properties_dict=properties_dict,
        cutoff=8.0, max_neighbors=12, use_zval=True, use_ionization_energy=True,
        use_cn=True, use_znorm=True, use_triplet_angle=False, use_surface=False
    )
    n = len(dataset)


    valid_data_list = []

    for i in range(n):
        data = dataset[i]
        try:
            ads = data.adsorption_energy if hasattr(data, 'adsorption_energy') else None

            if ads is not None:
                if isinstance(ads, torch.Tensor): val = ads.flatten()[0].item()
                elif isinstance(ads, (list, tuple)): val = float(ads[0])
                elif hasattr(ads, 'item'): val = ads.item()
                else: val = float(ads)

                if not math.isnan(val):
                    data.clean_target = torch.tensor([val], dtype=torch.float32)
                    data.sample_name = os.path.basename(cif_files[i])
                    valid_data_list.append(data)
        except Exception:
            pass

    num_valid = len(valid_data_list)
    if num_valid < 4:
        raise ValueError(f"Only {num_valid} samples have valid adsorption-energy labels; a 70/15/15 split is not possible. Check CSV columns and cache.")
    print(f"\n💡 Data filtering complete: {num_valid} samples have valid adsorption-energy labels.")

    metal_labels = [ (data.atomic_numbers[data.atomic_numbers >= 21][0].item() if len(data.atomic_numbers[data.atomic_numbers >= 21]) > 0 else 0) for data in valid_data_list ]

    from collections import Counter
    counts = Counter(metal_labels)
    metal_labels = [l if counts[l] >= 4 else -1 for l in metal_labels]


    if (args.train_csv is None) ^ (args.test_csv is None):
        raise ValueError("For a fixed split, provide both --train-csv and --test-csv.")

    if args.train_csv is not None and args.test_csv is not None:
        train_idx, val_idx, test_idx, norm_names = make_split_from_prediction_csv(
            valid_data_list, args.train_csv, args.test_csv
        )
    else:
        print(f"\n" + "="*40)
        print(f"⚡ Splitting the filtered dataset (Seed: {args.seed})...")
        try:
            train_idx, temp_idx, train_labels, temp_labels = train_test_split(
                np.arange(num_valid), metal_labels, test_size=0.30, stratify=metal_labels, random_state=args.seed
            )
        except ValueError:
            train_idx, temp_idx = train_test_split(
                np.arange(num_valid), test_size=0.30, random_state=args.seed
            )
            temp_labels = [metal_labels[i] for i in temp_idx]

        try:
            val_idx, test_idx, _, _ = train_test_split(
                temp_idx, temp_labels, test_size=0.50, stratify=temp_labels, random_state=args.seed
            )
        except ValueError:
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.50, random_state=args.seed
            )
        norm_names = [normalize_sample_name(getattr(d, "sample_name", f"sample_{i}")) for i, d in enumerate(valid_data_list)]

    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("At least one of the train, validation, or test sets is empty. Check labels and external split CSV files.")
    print(f"   [Sample split] Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print("="*40)

    split_rows = []
    split_map = {}
    for i in train_idx:
        split_map[int(i)] = "train"
    for i in val_idx:
        split_map[int(i)] = "val"
    for i in test_idx:
        split_map[int(i)] = "test"
    for i, data in enumerate(valid_data_list):
        split_rows.append({
            "index_in_valid_dataset": i,
            "Sample_Name": getattr(data, "sample_name", f"sample_{i}"),
            "normalized_name": norm_names[i],
            "split": split_map.get(i, "unused"),
        })
    pd.DataFrame(split_rows).to_csv(base_save_dir / "split_assignments.csv", index=False)
    print(f"📁 Split assignments saved to: {base_save_dir / 'split_assignments.csv'}")

    train_loader = DataLoader(torch.utils.data.Subset(valid_data_list, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(valid_data_list, val_idx), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(valid_data_list, test_idx), batch_size=args.batch_size, shuffle=False)

    model = DirectAdsorptionEnergyPredictor(
        node_dim=64, edge_dim=32, n_conv=3,
        atom_fea_len=int(dataset.atom_fea_len), edge_fea_len=int(dataset.edge_fea_len),
        use_triplet_angle=False
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = base_save_dir / "best_direct_model.pt"


    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred_ads = model(batch)
            loss = criterion(pred_ads.view(-1), batch.clean_target.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_ads = model(batch)
                val_loss += criterion(pred_ads.view(-1), batch.clean_target.view(-1)).item() * batch.num_graphs
        val_loss /= len(val_idx)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   [Epoch {epoch+1:03d}] Train Loss: {train_loss/len(train_idx):.4f} | Val Loss: {val_loss:.4f}")

        if patience_counter >= args.patience:
            print(f"   🛑 Early stopping at Epoch {epoch+1}: validation loss did not improve for {args.patience} consecutive epochs")
            break


    print("\n" + "🔍" * 30)
    print("Loading the checkpoint with the best validation loss and evaluating Test and Train sets...")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    def evaluate_and_save(loader, split_name):
        true_ads_list, pred_ads_list = [], []
        results_list = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred_ads = model(batch)

                t_vals = batch.clean_target.view(-1).cpu().numpy().tolist()
                p_vals = pred_ads.view(-1).cpu().numpy().tolist()

                true_ads_list.extend(t_vals)
                pred_ads_list.extend(p_vals)


                for i in range(len(t_vals)):
                    s_name = batch.sample_name[i] if hasattr(batch, 'sample_name') else f"Graph_{len(results_list)}"
                    results_list.append({
                        "Sample_Name": s_name,
                        "True_Adsorption_Energy": t_vals[i],
                        "Pred_Adsorption_Energy": p_vals[i],
                        "Absolute_Error": abs(p_vals[i] - t_vals[i])
                    })


        r2 = r2_score(true_ads_list, pred_ads_list)
        mae = mean_absolute_error(true_ads_list, pred_ads_list)
        rmse = np.sqrt(mean_squared_error(true_ads_list, pred_ads_list))

        print(f"\n🎉 Structure-only model results [{split_name.upper()}]:")
        print(f"   - R²   : {r2:.4f}")
        print(f"   - MAE  : {mae:.4f}")
        print(f"   - RMSE : {rmse:.4f}")


        df_results = pd.DataFrame(results_list)
        df_results.sort_values(by="Absolute_Error", ascending=False, inplace=True)
        df_results.to_csv(base_save_dir / f"{split_name}_predictions.csv", index=False)


        df_parity = df_results[["Sample_Name", "True_Adsorption_Energy", "Pred_Adsorption_Energy"]].copy()
        df_parity.sort_values(by="Sample_Name", inplace=True)
        df_parity.to_csv(base_save_dir / f"{split_name}_parity_plot_data.csv", index=False)


        try:
            plt.figure(figsize=(6, 5))
            plt.scatter(true_ads_list, pred_ads_list, alpha=0.6, edgecolors='w', s=50, c='steelblue')
            min_val = min(min(true_ads_list), min(pred_ads_list))
            max_val = max(max(true_ads_list), max(pred_ads_list))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            plt.title(f"Direct Structure Model - {split_name.capitalize()} (R²={r2:.4f}, MAE={mae:.4f})")
            plt.xlabel("True Adsorption Energy (eV)")
            plt.ylabel("Predicted Adsorption Energy (eV)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(base_save_dir / f"{split_name}_parity_plot.png", dpi=300)
            plt.close()
        except Exception as e:
            plt.close('all')
            print(f"[WARN] Failed to save {split_name} parity plot: {e}")

        print(f"📁 [{split_name.upper()}] Prediction table saved to: {base_save_dir / f'{split_name}_predictions.csv'}")


    evaluate_and_save(test_loader, "test")
    evaluate_and_save(train_loader, "train")

    print("\n" + "🏆" * 30)
    print(f"✅ Model checkpoint saved to:\n   👉 {best_model_path}")
    print("🏆" * 30)

if __name__ == "__main__":
    main()
