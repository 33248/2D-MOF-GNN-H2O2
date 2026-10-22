from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

import matplotlib
import inspect
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit, train_test_split

from hierarchical_cgcnn_triplet_angle import HierarchicalCGCNN, HierarchicalLoss
from data_loader_triplet_angle import MOFDataset

ELECTRONIC_KEYS = ["work_function", "d_band_center"]
ADSORPTION_COLUMNS = ("adsorption_energy", "adsorption_free_energy", "\u5438\u9644\u80fd", "\u5438\u9644\u81ea\u7531\u80fd", "ΔG*OOH", "ΔG_OOH", "dG_OOH")


def _filter_kwargs_for_callable(callable_obj, kwargs):

    try:
        sig = inspect.signature(callable_obj)
    except Exception:
        return kwargs
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def compute_norm_stats(
    cif_files: List[str],
    properties_dict: Dict[str, Dict],
    train_indices: List[int],
    keys: List[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals: List[float] = []
        for idx in train_indices:
            cif_name = os.path.basename(cif_files[idx])
            props = properties_dict.get(cif_name, {})
            if k in props:
                v = props[k]
                if v == v:
                    vals.append(float(v))
        if len(vals) == 0:
            continue
        arr = np.asarray(vals, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        out[k] = {"mean": mean, "std": std}
    return out


def has_valid_adsorption_energy_label(cif_idx: int, cif_files: List[str], properties_dict: Dict[str, Dict]) -> bool:

    cif_name = os.path.basename(cif_files[int(cif_idx)])
    props = properties_dict.get(cif_name, {})
    if "adsorption_energy" not in props:
        return False
    try:
        return bool(np.isfinite(float(props["adsorption_energy"])))
    except Exception:
        return False


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def compute_metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = compute_mse(y_true, y_pred)
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")
    return {"mse": mse, "r2": r2}


def load_electronic_weights(model: HierarchicalCGCNN, ckpt_path: str, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    filtered = {k: v for k, v in state_dict.items() if k.startswith("electronic_predictor.")}
    if not filtered:
        raise ValueError(f"No electronic_predictor.* parameters found in checkpoint: {ckpt_path}")

    current = model.state_dict()
    current.update(filtered)
    model.load_state_dict(current, strict=False)


def read_dataset_from_csv(data_folder: Path, task: str) -> Tuple[List[str], Dict[str, Dict]]:
    import pandas as pd

    csv_file = data_folder / "training_data.csv"
    if not csv_file.exists():
        csv_file = data_folder / "data.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {data_folder}/training_data.csv or {data_folder}/data.csv")

    df = pd.read_csv(csv_file)

    adsorption_energy_col = next((col for col in ADSORPTION_COLUMNS if col in df.columns), None)
    if adsorption_energy_col is not None:
        df[adsorption_energy_col] = pd.to_numeric(df[adsorption_energy_col], errors="coerce")

    if task == "electronic":
        missing = [c for c in ELECTRONIC_KEYS if c not in df.columns]
        if missing:
            raise ValueError(f"--task electronic requires the following CSV columns: {missing}, but they were not found.")

    if task == "joint":
        if adsorption_energy_col is None:
            raise ValueError(f"--task joint requires an adsorption-energy column. Supported names: {', '.join(ADSORPTION_COLUMNS)}")

    cif_files: List[str] = []
    properties_dict: Dict[str, Dict] = {}

    def _normalize_filename(name: str) -> str:
        name = str(name).strip()
        if not name:
            return name
        if not name.lower().endswith(".cif"):
            name = name + ".cif"
        return name

    for _, row in df.iterrows():
        cif_path: Optional[str] = None

        if "cif_file" in df.columns:
            cif_path = str(row["cif_file"]).strip()

        if (cif_path is None or cif_path == "") and "cif_filename" in df.columns:
            raw = str(row["cif_filename"]).strip()
            if raw:
                if os.path.exists(raw):
                    cif_path = raw
                else:
                    cand1 = str((data_folder / raw).resolve())
                    if os.path.exists(cand1):
                        cif_path = cand1
                    else:
                        cand2 = str((data_folder / "cif_files" / _normalize_filename(raw)).resolve())
                        cif_path = cand2

        if (cif_path is None or cif_path == "") and "cif\u6587\u4ef6\u540d" in df.columns:
            raw = str(row["cif\u6587\u4ef6\u540d"]).strip()
            raw = _normalize_filename(raw)
            cif_path = str((data_folder / "cif_files" / raw).resolve())

        if cif_path is None or cif_path == "":
            raise ValueError("No usable CIF column in CSV. Provide cif_file, cif_filename, or cif\u6587\u4ef6\u540d")

        if not os.path.exists(cif_path):
            raise FileNotFoundError(
                f"CIF file not found: {cif_path}. Check the path/filename in the CSV, "
                f"or place the CIF in {data_folder}/cif_files/."
            )

        cif_files.append(cif_path)
        cif_name = os.path.basename(cif_path)
        props: Dict[str, float] = {}


        if task == "joint" and adsorption_energy_col is not None:
            val = row[adsorption_energy_col]
            if pd.notna(val) and np.isfinite(val):
                props["adsorption_energy"] = float(val)

        for k in ELECTRONIC_KEYS:
            if k in df.columns:
                val = row[k]
                if val == val:
                    props[k] = float(val)

        properties_dict[cif_name] = props

    return cif_files, properties_dict


class Trainer:
    def __init__(
        self,
        model: HierarchicalCGCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        task: str,
        device: str,
        learning_rate: float,
        save_dir: str,
        lambda_electronic: float,
        lambda_adsorption_energy: float,
        lambda_constraint: float,
        freeze_electronic_epochs: int = 0,
        norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.task = task
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if task == "electronic":
            lambda_adsorption_energy = 0.0

        self.norm_stats = norm_stats or {}
        self.criterion = HierarchicalLoss(
            lambda_electronic=lambda_electronic,
            lambda_adsorption_energy=lambda_adsorption_energy,
            lambda_constraint=lambda_constraint,
            norm_stats=self.norm_stats,
        )

        if task == "electronic":
            for p in self.model.adsorption_energy_predictor.parameters():
                p.requires_grad = False

        self.optimizer = Adam([p for p in self.model.parameters() if p.requires_grad], lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=10)

        self.freeze_electronic_epochs = max(0, int(freeze_electronic_epochs))

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
        }
        if task == "electronic":
            self.history.update({
                "train_e_avg_mse": [], "val_e_avg_mse": [],
                "train_e_avg_r2": [], "val_e_avg_r2": [],
            })
        else:
            self.history.update({
                "train_adsorption_mae": [], "val_adsorption_mae": [],
            })

        self.best_val_loss = float("inf")

    def _set_electronic_trainable(self, trainable: bool) -> None:
        for p in self.model.electronic_predictor.parameters():
            p.requires_grad = trainable

    def _step_batch_electronic(self, batch) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, np.ndarray]]:
        electronic_features, _ = self.model.electronic_predictor(batch)
        predictions = {"electronic_features": electronic_features}
        targets = {k: getattr(batch, k) for k in ELECTRONIC_KEYS}

        loss, _ = self.criterion(predictions, targets)
        preds_targets: Dict[str, np.ndarray] = {}
        for k in ELECTRONIC_KEYS:
            preds_targets[f"pred_{k}"] = electronic_features[k].detach().cpu().numpy().reshape(-1)
            preds_targets[f"true_{k}"] = getattr(batch, k).detach().cpu().numpy().reshape(-1)

        mses, r2s = [], []
        for k in ELECTRONIC_KEYS:
            m = compute_metrics_regression(preds_targets[f"true_{k}"], preds_targets[f"pred_{k}"])
            mses.append(m["mse"])
            r2s.append(m["r2"])
        metrics = {
            "avg_mse": float(np.mean(mses)),
            "avg_r2": float(np.nanmean(r2s)),
        }
        return loss, metrics, preds_targets

    def _step_batch_joint(self, batch) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        adsorption_energy_pred, electronic_features = self.model(batch, return_electronic=True)
        predictions = {
            "adsorption_energy": adsorption_energy_pred,
            "electronic_features": electronic_features,
        }

        targets: Dict[str, torch.Tensor] = {}
        if hasattr(batch, "y"):
            targets["adsorption_energy"] = batch.y
        for k in ELECTRONIC_KEYS:
            if hasattr(batch, k):
                targets[k] = getattr(batch, k)

        loss, _ = self.criterion(predictions, targets)

        out: Dict[str, np.ndarray] = {
            "pred_adsorption": adsorption_energy_pred.detach().cpu().numpy().reshape(-1),
            "true_adsorption": batch.y.detach().cpu().numpy().reshape(-1),
        }
        for k in ELECTRONIC_KEYS:
            if k in electronic_features and hasattr(batch, k):
                out[f"pred_{k}"] = electronic_features[k].detach().cpu().numpy().reshape(-1)
                out[f"true_{k}"] = getattr(batch, k).detach().cpu().numpy().reshape(-1)

        return loss, out

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0

        if self.task == "electronic":
            all_true: Dict[str, List[float]] = {k: [] for k in ELECTRONIC_KEYS}
            all_pred: Dict[str, List[float]] = {k: [] for k in ELECTRONIC_KEYS}

            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                loss, _, preds_targets = self._step_batch_electronic(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += float(loss.item())
                for k in ELECTRONIC_KEYS:
                    all_true[k].extend(preds_targets[f"true_{k}"].tolist())
                    all_pred[k].extend(preds_targets[f"pred_{k}"].tolist())

            avg_loss = total_loss / max(1, len(self.train_loader))
            per_key = {k: compute_metrics_regression(all_true[k], all_pred[k]) for k in ELECTRONIC_KEYS}
            avg_mse = float(np.mean([per_key[k]["mse"] for k in ELECTRONIC_KEYS]))
            avg_r2 = float(np.nanmean([per_key[k]["r2"] for k in ELECTRONIC_KEYS]))

            return {"loss": avg_loss, "e_avg_mse": avg_mse, "e_avg_r2": avg_r2, **{f"e_{k}_mse": per_key[k]["mse"] for k in ELECTRONIC_KEYS}, **{f"e_{k}_r2": per_key[k]["r2"] for k in ELECTRONIC_KEYS}}

        all_adsorption_true: List[float] = []
        all_adsorption_pred: List[float] = []
        all_true: Dict[str, List[float]] = {k: [] for k in ELECTRONIC_KEYS}
        all_pred: Dict[str, List[float]] = {k: [] for k in ELECTRONIC_KEYS}

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            loss, out = self._step_batch_joint(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            all_adsorption_true.extend(out["true_adsorption"].tolist())
            all_adsorption_pred.extend(out["pred_adsorption"].tolist())
            for k in ELECTRONIC_KEYS:
                if f"true_{k}" in out:
                    all_true[k].extend(out[f"true_{k}"].tolist())
                    all_pred[k].extend(out[f"pred_{k}"].tolist())

        avg_loss = total_loss / max(1, len(self.train_loader))
        adsorption_mae = float(mean_absolute_error(all_adsorption_true, all_adsorption_pred))
        adsorption_metrics = compute_metrics_regression(all_adsorption_true, all_adsorption_pred)

        per_key = {k: compute_metrics_regression(all_true[k], all_pred[k]) for k in ELECTRONIC_KEYS if len(all_true[k]) > 0}
        avg_e_mse = float(np.mean([per_key[k]["mse"] for k in per_key])) if per_key else float("nan")
        avg_e_r2 = float(np.nanmean([per_key[k]["r2"] for k in per_key])) if per_key else float("nan")

        out_metrics = {
            "loss": avg_loss,
            "adsorption_mae": adsorption_mae,
            "adsorption_mse": adsorption_metrics["mse"],
            "adsorption_r2": adsorption_metrics["r2"],
            "e_avg_mse": avg_e_mse,
            "e_avg_r2": avg_e_r2,
        }
        for k in per_key:
            out_metrics[f"e_{k}_mse"] = per_key[k]["mse"]
            out_metrics[f"e_{k}_r2"] = per_key[k]["r2"]
        return out_metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[Dict[str, float], Dict[str, Dict[str, List[float]]], List[str]]:
        self.model.eval()
        total_loss = 0.0
        pred_true: Dict[str, Dict[str, List[float]]] = {}
        sample_ids: List[str] = []

        if self.task == "electronic":
            for k in ELECTRONIC_KEYS:
                pred_true[k] = {"pred": [], "true": []}

            for batch in loader:
                batch = batch.to(self.device)
                loss, _, preds_targets = self._step_batch_electronic(batch)
                total_loss += float(loss.item())

                if hasattr(batch, "sample_id"):
                    bid = batch.sample_id
                    if isinstance(bid, (list, tuple)):
                        sample_ids.extend([str(x) for x in bid])
                    else:
                        sample_ids.append(str(bid))
                else:
                    n_graphs = int(preds_targets[f"pred_{ELECTRONIC_KEYS[0]}"].shape[0])
                    sample_ids.extend([f"sample_{len(sample_ids)+i}" for i in range(n_graphs)])

                for k in ELECTRONIC_KEYS:
                    pred_true[k]["pred"].extend(preds_targets[f"pred_{k}"].tolist())
                    pred_true[k]["true"].extend(preds_targets[f"true_{k}"].tolist())

            avg_loss = total_loss / max(1, len(loader))
            per_key = {k: compute_metrics_regression(pred_true[k]["true"], pred_true[k]["pred"]) for k in ELECTRONIC_KEYS}
            avg_mse = float(np.mean([per_key[k]["mse"] for k in ELECTRONIC_KEYS]))
            avg_r2 = float(np.nanmean([per_key[k]["r2"] for k in ELECTRONIC_KEYS]))

            metrics = {"loss": avg_loss, "e_avg_mse": avg_mse, "e_avg_r2": avg_r2}
            for k in ELECTRONIC_KEYS:
                metrics[f"e_{k}_mse"] = per_key[k]["mse"]
                metrics[f"e_{k}_r2"] = per_key[k]["r2"]
            return metrics, pred_true, sample_ids

        pred_true["adsorption_energy"] = {"pred": [], "true": []}
        for k in ELECTRONIC_KEYS:
            pred_true[k] = {"pred": [], "true": []}

        for batch in loader:
            batch = batch.to(self.device)
            loss, out = self._step_batch_joint(batch)
            total_loss += float(loss.item())

            if hasattr(batch, "sample_id"):
                bid = batch.sample_id
                if isinstance(bid, (list, tuple)):
                    sample_ids.extend([str(x) for x in bid])
                else:
                    sample_ids.append(str(bid))
            else:
                n_graphs = int(out["pred_adsorption"].shape[0])
                sample_ids.extend([f"sample_{len(sample_ids)+i}" for i in range(n_graphs)])

            pred_true["adsorption_energy"]["pred"].extend(out["pred_adsorption"].tolist())
            pred_true["adsorption_energy"]["true"].extend(out["true_adsorption"].tolist())

            for k in ELECTRONIC_KEYS:
                if f"true_{k}" in out:
                    pred_true[k]["pred"].extend(out[f"pred_{k}"].tolist())
                    pred_true[k]["true"].extend(out[f"true_{k}"].tolist())

        avg_loss = total_loss / max(1, len(loader))
        adsorption_mae = float(mean_absolute_error(pred_true["adsorption_energy"]["true"], pred_true["adsorption_energy"]["pred"]))
        adsorption_metrics = compute_metrics_regression(pred_true["adsorption_energy"]["true"], pred_true["adsorption_energy"]["pred"])

        per_key = {k: compute_metrics_regression(pred_true[k]["true"], pred_true[k]["pred"]) for k in ELECTRONIC_KEYS}
        avg_e_mse = float(np.mean([per_key[k]["mse"] for k in ELECTRONIC_KEYS]))
        avg_e_r2 = float(np.nanmean([per_key[k]["r2"] for k in ELECTRONIC_KEYS]))

        metrics = {
            "loss": avg_loss,
            "adsorption_mae": adsorption_mae,
            "adsorption_mse": adsorption_metrics["mse"],
            "adsorption_r2": adsorption_metrics["r2"],
            "e_avg_mse": avg_e_mse,
            "e_avg_r2": avg_e_r2,
        }
        for k in ELECTRONIC_KEYS:
            metrics[f"e_{k}_mse"] = per_key[k]["mse"]
            metrics[f"e_{k}_r2"] = per_key[k]["r2"]

        return metrics, pred_true, sample_ids

    def save_checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        ckpt = {
            "epoch": epoch,
            "task": self.task,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
            "norm_stats": self.norm_stats,
        }
        torch.save(ckpt, self.save_dir / filename)

    def load_checkpoint(self, filename: str) -> None:
        ckpt = torch.load(self.save_dir / filename, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.history = ckpt.get("history", self.history)

    def export_predictions_and_outliers(
        self,
        pred_true: Dict[str, Dict[str, List[float]]],
        sample_ids: List[str],
        keys_to_export: List[str],
        top_n: int = 30,
        prefix: str = "",
    ) -> None:
        if not sample_ids or not keys_to_export:
            return
        first_key = keys_to_export[0]
        n = len(pred_true[first_key]["true"])
        if len(sample_ids) != n:
            print(f"[WARN] sample_ids length ({len(sample_ids)}) != n_preds ({n}); skip export.")
            return
        try:
            import pandas as pd
            rows = []
            for idx, sid in enumerate(sample_ids):
                sid = str(sid)
                parts = sid.split("-", 1)
                metal = parts[0] if len(parts) >= 1 else ""
                ligand = parts[1] if len(parts) == 2 else ""
                row = {"sample_id": sid, "metal": metal, "ligand": ligand}
                abs_sum = 0.0
                for k in keys_to_export:
                    t = float(pred_true[k]["true"][idx])
                    p = float(pred_true[k]["pred"][idx])
                    err = p - t
                    row[f"true_{k}"] = t
                    row[f"pred_{k}"] = p
                    row[f"err_{k}"] = err
                    row[f"abs_err_{k}"] = abs(err)
                    abs_sum += abs(err)
                row["abs_err_sum"] = float(abs_sum)
                rows.append(row)
            dfp = pd.DataFrame(rows)

            dfp.to_csv(self.save_dir / f"{prefix}predictions.csv", index=False)
            top = dfp.sort_values("abs_err_sum", ascending=False).head(top_n)
            top.to_csv(self.save_dir / f"{prefix}top_outliers.csv", index=False)
            for k in keys_to_export:
                col = f"abs_err_{k}"
                if col in dfp.columns:
                    dfk = dfp.sort_values(col, ascending=False).head(top_n)
                    dfk.to_csv(self.save_dir / f"{prefix}top_outliers_{k}.csv", index=False)
        except Exception as e:
            print(f"[WARN] Failed to export predictions/outliers CSV: {e}")

    def plot_training_curves(self) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(self.history["train_loss"], label="train_loss")
        ax.plot(self.history["val_loss"], label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss ({self.task})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.save_dir / "training_loss.png"
        plt.savefig(out, dpi=300)
        plt.close()

        if self.task == "electronic":
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(self.history["train_e_avg_mse"], label="train")
            ax[0].plot(self.history["val_e_avg_mse"], label="val")
            ax[0].set_title("Electronic avg MSE")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("MSE")
            ax[0].legend(); ax[0].grid(True, alpha=0.3)

            ax[1].plot(self.history["train_e_avg_r2"], label="train")
            ax[1].plot(self.history["val_e_avg_r2"], label="val")
            ax[1].set_title("Electronic avg R²")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("R²")
            ax[1].legend(); ax[1].grid(True, alpha=0.3)

            plt.tight_layout()
            out = self.save_dir / "training_electronic_metrics.png"
            plt.savefig(out, dpi=300)
            plt.close()
        else:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(self.history["train_adsorption_mae"], label="train")
            ax[0].plot(self.history["val_adsorption_mae"], label="val")
            ax[0].set_title("Adsorption Energy MAE")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("MAE")
            ax[0].legend(); ax[0].grid(True, alpha=0.3)

            ax[1].plot(self.history.get("train_loss", []), label="train_loss")
            ax[1].plot(self.history.get("val_loss", []), label="val_loss")
            ax[1].set_title("Loss")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Loss")
            ax[1].legend(); ax[1].grid(True, alpha=0.3)

            plt.tight_layout()
            out = self.save_dir / "training_adsorption_energy_metrics.png"
            plt.savefig(out, dpi=300)
            plt.close()

    def plot_predictions(self, pred_true: Dict[str, Dict[str, List[float]]], metrics: Dict[str, float], sample_ids: Optional[List[str]] = None) -> None:
        keys_to_plot = ["adsorption_energy"] if (self.task == "joint") else []
        keys_to_plot += ELECTRONIC_KEYS

        n = len(keys_to_plot)
        if n == 0:
            return
        n_cols = 2
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = np.array(axes).reshape(-1)

        for i, k in enumerate(keys_to_plot):
            ax = axes[i]
            y_true = np.asarray(pred_true[k]["true"]).reshape(-1)
            y_pred = np.asarray(pred_true[k]["pred"]).reshape(-1)

            ax.scatter(y_true, y_pred, alpha=0.6, s=25)
            mn = float(min(y_true.min(), y_pred.min()))
            mx = float(max(y_true.max(), y_pred.max()))
            ax.plot([mn, mx], [mn, mx], "r--", lw=2)

            m = compute_metrics_regression(y_true, y_pred)
            title = f"{k}\nMSE={m['mse']:.4g}, R²={m['r2']:.4g}"
            if k == "adsorption_energy":
                title += f", MAE={float(mean_absolute_error(y_true, y_pred)):.4g}"

            ax.set_title(title)
            ax.set_xlabel("True")
            ax.set_ylabel("Pred")
            ax.grid(True, alpha=0.3)

        for j in range(n, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        out = self.save_dir / "predictions.png"
        plt.savefig(out, dpi=300)
        plt.close()

    def train(self, n_epochs: int, early_stopping_patience: int) -> Dict[str, Dict[str, float]]:
        print("=" * 60)
        print(f"Start Training | task={self.task}")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Train/Val/Test: {len(self.train_loader.dataset)}/{len(self.val_loader.dataset)}/{len(self.test_loader.dataset)}")
        if self.task == "joint" and self.freeze_electronic_epochs > 0:
            print(f"Freeze electronic predictor for first {self.freeze_electronic_epochs} epochs")
        print("=" * 60)

        patience = 0

        for epoch in range(n_epochs):
            if self.task == "joint" and self.freeze_electronic_epochs > 0:
                self._set_electronic_trainable(epoch >= self.freeze_electronic_epochs)

            train_metrics = self.train_epoch()
            val_metrics, _, _ = self.evaluate(self.val_loader)

            self.scheduler.step(val_metrics["loss"])

            self.history["train_loss"].append(float(train_metrics["loss"]))
            self.history["val_loss"].append(float(val_metrics["loss"]))

            if self.task == "electronic":
                self.history["train_e_avg_mse"].append(float(train_metrics["e_avg_mse"]))
                self.history["val_e_avg_mse"].append(float(val_metrics["e_avg_mse"]))
                self.history["train_e_avg_r2"].append(float(train_metrics["e_avg_r2"]))
                self.history["val_e_avg_r2"].append(float(val_metrics["e_avg_r2"]))
            else:
                self.history["train_adsorption_mae"].append(float(train_metrics["adsorption_mae"]))
                self.history["val_adsorption_mae"].append(float(val_metrics["adsorption_mae"]))

            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            if self.task == "electronic":
                print(f"  Train Loss: {train_metrics['loss']:.4f} | e_avg_mse: {train_metrics['e_avg_mse']:.4g} | e_avg_r2: {train_metrics['e_avg_r2']:.4g}")
                print(f"  Val   Loss: {val_metrics['loss']:.4f} | e_avg_mse: {val_metrics['e_avg_mse']:.4g} | e_avg_r2: {val_metrics['e_avg_r2']:.4g}")
                for k in ELECTRONIC_KEYS:
                    print(f"    {k}: MSE={val_metrics[f'e_{k}_mse']:.4g}, R²={val_metrics[f'e_{k}_r2']:.4g}")
            else:
                print(f"  Train Loss: {train_metrics['loss']:.4f} | adsorption_mae: {train_metrics['adsorption_mae']:.4g} | adsorption_r2: {train_metrics['adsorption_r2']:.4g}")
                print(f"  Val   Loss: {val_metrics['loss']:.4f} | adsorption_mae: {val_metrics['adsorption_mae']:.4g} | adsorption_r2: {val_metrics['adsorption_r2']:.4g}")
                print(f"  Val Electronic: e_avg_mse={val_metrics['e_avg_mse']:.4g}, e_avg_r2={val_metrics['e_avg_r2']:.4g}")

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = float(val_metrics["loss"])
                self.save_checkpoint("best_model.pt", epoch, self.best_val_loss)
                patience = 0
                print(f"  ✓ Saved best_model.pt (val_loss={self.best_val_loss:.4f})")
            else:
                patience += 1

            if patience >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        print("\n" + "=" * 60)
        print("Testing best checkpoint...")
        print("=" * 60)
        self.load_checkpoint("best_model.pt")

        test_metrics, pred_true, sample_ids = self.evaluate(self.test_loader)
        print("\n" + "=" * 60)
        print("Final Test Results")
        print("=" * 60)

        if self.task == "electronic":
            print(f"Electronic Avg MSE: {test_metrics['e_avg_mse']:.6f}")
            print(f"Electronic Avg R² : {test_metrics['e_avg_r2']:.6f}")
            for k in ELECTRONIC_KEYS:
                print(f"{k}:")
                print(f"  MSE = {test_metrics[f'e_{k}_mse']:.6f}")
                print(f"  R²  = {test_metrics[f'e_{k}_r2']:.6f}")
        else:
            print("Adsorption Energy Prediction:")
            print(f"  MAE = {test_metrics['adsorption_mae']:.6f}")
            print(f"  MSE = {test_metrics['adsorption_mse']:.6f}")
            print(f"  R²  = {test_metrics['adsorption_r2']:.6f}")
            print("\nElectronic Properties:")
            print(f"  Electronic Avg MSE = {test_metrics['e_avg_mse']:.6f}")
            print(f"  Electronic Avg R²  = {test_metrics['e_avg_r2']:.6f}")
            for k in ELECTRONIC_KEYS:
                print(f"{k}:")
                print(f"  MSE = {test_metrics[f'e_{k}_mse']:.6f}")
                print(f"  R²  = {test_metrics[f'e_{k}_r2']:.6f}")
        print("=" * 60)

        keys_to_export = ELECTRONIC_KEYS if (self.task == "electronic") else (["adsorption_energy"] + ELECTRONIC_KEYS)


        self.export_predictions_and_outliers(
            pred_true=pred_true,
            sample_ids=sample_ids,
            keys_to_export=keys_to_export,
            top_n=30,
            prefix="test_"
        )


        print("\n" + "=" * 60)
        print("Evaluating best checkpoint on Train Set...")
        print("=" * 60)
        train_metrics_final, train_pred_true, train_sample_ids = self.evaluate(self.train_loader)

        self.export_predictions_and_outliers(
            pred_true=train_pred_true,
            sample_ids=train_sample_ids,
            keys_to_export=keys_to_export,
            top_n=30,
            prefix="train_"
        )

        with open(self.save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        with open(self.save_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

        self.plot_training_curves()
        self.plot_predictions(pred_true, test_metrics, sample_ids=sample_ids)

        results: Dict[str, Dict[str, float]] = {}
        if self.task == "electronic":
            for k in ELECTRONIC_KEYS:
                results[k] = {"mse": float(test_metrics[f"e_{k}_mse"]), "r2": float(test_metrics[f"e_{k}_r2"])}
            results["electronic_avg"] = {"mse": float(test_metrics["e_avg_mse"]), "r2": float(test_metrics["e_avg_r2"])}
        else:
            results["adsorption_energy"] = {
                "mae": float(test_metrics["adsorption_mae"]), "mse": float(test_metrics["adsorption_mse"]), "r2": float(test_metrics["adsorption_r2"])
            }
            for k in ELECTRONIC_KEYS:
                results[k] = {"mse": float(test_metrics[f"e_{k}_mse"]), "r2": float(test_metrics[f"e_{k}_r2"])}
            results["electronic_avg"] = {"mse": float(test_metrics["e_avg_mse"]), "r2": float(test_metrics["e_avg_r2"])}

        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hierarchical CGCNN (electronic-only or joint)")
    parser.add_argument(
        "data_path",
        type=str,
        nargs="?",
        default="training_dataset/2/",
        help="Path to the data directory containing CSV and CIF files",
    )
    parser.add_argument("--task", choices=["electronic", "joint"], default="joint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Delete the <data_path>/processed cache and rebuild graphs from CIF files",
    )
    parser.add_argument(
        "--no-label-norm",
        action="store_true",
        help="Disable target normalization",
    )
    parser.add_argument("--atom-init", type=str, default=None)
    parser.add_argument("--use-zval", action="store_true")
    parser.add_argument("--zval-map", type=str, default=None)
    parser.add_argument("--zval-normalize", choices=["max", "none"], default="max")
    parser.add_argument("--use-ionization-energy", action="store_true")
    parser.add_argument("--ionization-map", type=str, default=None)
    parser.add_argument("--ionization-normalize", choices=["max", "none"], default="max")
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--rbf-bins", type=int, default=64)
    parser.add_argument("--no-rbf", action="store_true")
    parser.add_argument("--use-cn", action="store_true", default=True)
    parser.add_argument("--use-surface", action="store_true")
    parser.add_argument("--use-znorm", action="store_true")
    parser.add_argument("--use-triplet-angle", action="store_true")
    parser.add_argument("--angle-rbf-bins", type=int, default=32)
    parser.add_argument("--angle-gamma", type=float, default=10.0)
    parser.add_argument("--max-triplets-per-center", type=int, default=64)
    parser.add_argument("--lambda-electronic", type=float, default=0.3)
    parser.add_argument("--lambda-adsorption-energy", type=float, default=1.0)
    parser.add_argument("--lambda-constraint", type=float, default=0.1)
    parser.add_argument("--init-electronic-ckpt", type=str, default=None)
    parser.add_argument("--freeze-electronic-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)


    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("HierarchicalCGCNN Training - pre-filtered single split (70:15:15)")
    print("=" * 60)
    print(f"task={args.task} | device={device} | seed={args.seed}")

    set_seed(args.seed)
    data_folder = Path(args.data_path)

    try:
        cif_files, properties_dict = read_dataset_from_csv(data_folder, task=args.task)
    except Exception as e:
        print(f"Data loading failed: {e}")
        sys.exit(1)

    if args.reprocess:
        import shutil
        processed_dir = data_folder / "processed"
        if processed_dir.exists():
            print(f"--reprocess enabled: removing old cache: {processed_dir}")
            shutil.rmtree(processed_dir)

    dataset = MOFDataset(
        root=str(data_folder),
        cif_files=cif_files,
        properties_dict=properties_dict,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        atom_init_path=args.atom_init,
        use_rbf=not args.no_rbf,
        rbf_bins=args.rbf_bins,
        use_zval=args.use_zval,
        zval_map_path=args.zval_map,
        zval_normalize=args.zval_normalize,
        use_ionization_energy=args.use_ionization_energy,
        ionization_map_path=args.ionization_map,
        ionization_normalize=args.ionization_normalize,
        use_cn=args.use_cn,
        use_surface=args.use_surface,
        use_znorm=args.use_znorm,
        use_triplet_angle=args.use_triplet_angle,
        angle_rbf_bins=args.angle_rbf_bins,
        angle_gamma=args.angle_gamma,
        max_triplets_per_center=args.max_triplets_per_center,
    )

    base_save_dir = Path(args.save_dir)
    base_save_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(base_save_dir / "args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
    except Exception:
        pass

    def _maybe_reuse_electronic_stats(norm_stats: Dict[str, Dict[str, float]]) -> None:
        if not (args.task == "joint" and args.init_electronic_ckpt):
            return
        try:
            ckpt = torch.load(args.init_electronic_ckpt, map_location="cpu")
            ckpt_stats = ckpt.get("norm_stats", None)
            if isinstance(ckpt_stats, dict):
                for k in ELECTRONIC_KEYS:
                    if k in ckpt_stats:
                        norm_stats[k] = ckpt_stats[k]
                print("✓ Reused electronic norm_stats from init ckpt")
        except Exception:
            return

    def _apply_joint_pre_filter(all_idxs: List[int], save_dir: Path) -> List[int]:


        if args.task != "joint":
            return all_idxs

        original_count = len(all_idxs)
        kept, removed_empty_adsorption_energy, stale_cache, invalid_y = [], [], [], []

        def _names(idxs: List[int]) -> List[str]:
            return [os.path.basename(cif_files[int(i)]) for i in idxs]

        for idx in map(int, all_idxs):
            if not has_valid_adsorption_energy_label(idx, cif_files, properties_dict):
                removed_empty_adsorption_energy.append(idx)
                continue

            try:
                data_i = dataset[idx]
            except Exception as e:
                raise RuntimeError(f"Failed to load sample: idx={idx}, error={e}") from e

            if not hasattr(data_i, "y"):
                stale_cache.append(idx)
                continue

            try:
                yval = float(data_i.y.view(-1)[0].item())
                if not np.isfinite(yval):
                    invalid_y.append(idx)
                    continue
            except Exception:
                invalid_y.append(idx)
                continue

            kept.append(idx)


        report = {
            "strategy": "Pre-filter then split",
            "original_count": original_count,
            "kept_count": len(kept),
            "removed_empty_adsorption_energy_count": len(removed_empty_adsorption_energy),
            "stale_cache_missing_y_count": len(stale_cache),
            "invalid_y_count": len(invalid_y),
            "removed_empty_adsorption_energy_files": _names(removed_empty_adsorption_energy),
            "stale_cache_missing_y_files": _names(stale_cache),
            "invalid_y_files": _names(invalid_y),
        }

        print("=" * 60)
        print("Pre-filtering: selecting samples with valid adsorption-energy labels")
        print(f"  Original count: {original_count} -> Valid after filtering: {len(kept)}")
        print(f"  (Filtering details: missing labels {len(removed_empty_adsorption_energy)}, stale cache {len(stale_cache)}, invalid values {len(invalid_y)})")
        print("=" * 60)

        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "pre_filter_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        if stale_cache:
            raise RuntimeError(f"Graph data missing data.y for {len(stale_cache)} samples. Use --reprocess to clear stale cache files and retry.")

        if len(kept) < 3:
            raise RuntimeError("Too few valid samples (fewer than 3) remain after filtering. Check the adsorption-energy labels.")

        return kept

    def run_one_split(train_idx: List[int], val_idx: List[int], test_idx: List[int], save_dir: Path) -> Dict[str, Dict[str, float]]:
        norm_stats: Dict[str, Dict[str, float]] = {}
        if not args.no_label_norm:
            keys_for_norm = ELECTRONIC_KEYS.copy()
            if args.task == "joint":
                keys_for_norm = ["adsorption_energy"] + keys_for_norm
            norm_stats = compute_norm_stats(cif_files, properties_dict, train_idx, keys_for_norm)
            _maybe_reuse_electronic_stats(norm_stats)
            with open(save_dir / "label_stats.json", "w", encoding="utf-8") as f:
                json.dump(norm_stats, f, indent=2)

        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

        model_kwargs = dict(
            node_dim=64, edge_dim=32, global_dim=16, electronic_dim=128,
            atom_fea_len=int(dataset.atom_fea_len), edge_fea_len=int(dataset.edge_fea_len),
            use_triplet_angle=args.use_triplet_angle, angle_rbf_bins=args.angle_rbf_bins,
        )
        model = HierarchicalCGCNN(**_filter_kwargs_for_callable(HierarchicalCGCNN, model_kwargs))

        if args.task == "joint" and args.init_electronic_ckpt:
            ckpt_path = args.init_electronic_ckpt
            print(f"Loading electronic weights from: {ckpt_path}")
            load_electronic_weights(model, ckpt_path, device=device)

        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            task=args.task, device=device, learning_rate=args.lr, save_dir=str(save_dir),
            lambda_electronic=args.lambda_electronic, lambda_adsorption_energy=args.lambda_adsorption_energy,
            lambda_constraint=args.lambda_constraint, freeze_electronic_epochs=args.freeze_electronic_epochs,
            norm_stats=norm_stats,
        )
        results = trainer.train(n_epochs=args.epochs, early_stopping_patience=args.patience)
        with open(save_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return results


    n = len(dataset)
    all_idx = np.arange(n)


    pure_indices = _apply_joint_pre_filter(all_idx.tolist(), base_save_dir)
    pure_arr = np.array(pure_indices)

    print("\n" + "=" * 60)
    print(f"Random split: using {len(pure_arr)} valid samples for a 70:15:15 split")
    print("=" * 60)


    trainval_idx, test_idx = train_test_split(pure_arr, test_size=0.15, random_state=args.seed)


    val_ratio_adjusted = 0.15 / 0.85
    train_idx, val_idx = train_test_split(trainval_idx, test_size=val_ratio_adjusted, random_state=args.seed)

    train_idx = train_idx.tolist()
    val_idx = val_idx.tolist()
    test_idx = test_idx.tolist()

    print(f"  Split sizes -> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")


    results = run_one_split(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, save_dir=base_save_dir)

    print("\n" + "=" * 60)
    print("Done (Random Single Split)")
    print("=" * 60)
    print(f"Saved to: {args.save_dir}")

    if args.task == "electronic":
        print("Electronic test metrics (MSE / R²):")
        for k in ELECTRONIC_KEYS:
            print(f"  {k}: MSE={results[k]['mse']:.4g}, R²={results[k]['r2']:.4g}")
        print(f"  avg: MSE={results['electronic_avg']['mse']:.4g}, R²={results['electronic_avg']['r2']:.4g}")
    else:
        a = results["adsorption_energy"]
        print(f"adsorption_energy: MAE={a['mae']:.4g}, MSE={a['mse']:.4g}, R²={a['r2']:.4g}")
        print(f"Electronic avg: MSE={results['electronic_avg']['mse']:.4g}, R²={results['electronic_avg']['r2']:.4g}")

if __name__ == "__main__":
    main()
