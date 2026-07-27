#!/usr/bin/env python3

"""
Assess SIIM-ISIC 90-model ensemble on a Derm7pt dataset.

Example:
  python siim90_assess_derm7pt.py \
      --csv ../derm7pt/release_v0/meta/meta.csv \
      --image_dir ../derm7pt/release_v0/images \
      --model_dir /Users/qwang/models/melanoma-winning-models \
      --image_col derm \
      --diagnosis_col diagnosis \
      --output_csv siim90_predict_on_derm7pt/siim90_derm7pt_predictions.csv \
      --output_metrics siim90_predict_on_derm7pt/siim90_derm7pt_metrics.json 
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve
)

import siim90_predict_one as siim

def build_image_index(image_dir):
    image_dir = Path(image_dir)
    image_index = {}

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        for p in image_dir.rglob(ext):
            image_index[p.name] = p
            image_index[p.stem] = p

    return image_index


def resolve_image_path(value, image_dir, image_index):
    value = str(value).strip()
    p = Path(value)

    if p.is_absolute() and p.exists():
        return p

    candidate = Path(image_dir) / value
    if candidate.exists():
        return candidate

    if value in image_index:
        return image_index[value]

    if p.name in image_index:
        return image_index[p.name]

    if p.stem in image_index:
        return image_index[p.stem]

    # Try adding common extensions
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        if value + ext in image_index:
            return image_index[value + ext]

    return None
    
    
def get_derm7pt_labels(row, diagnosis_col):
    dx = str(row[diagnosis_col]).strip().lower()

    y_melanoma = int("melanoma" in dx)

    malignant = [
        "melanoma",
        "basal cell carcinoma",
        "squamous cell carcinoma",
    ]

    y_malignant = int(any(x in dx for x in malignant))

    return y_malignant, y_melanoma
    

def compute_metrics(y_true, y_prob, threshold=0.5, optimize_threshold=False):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if optimize_threshold:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden = tpr - fpr
        threshold = thresholds[np.argmax(youden)]

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "operating_point": {
            "n": int(len(y_true)),
            "threshold": float(threshold),
        },
        "performance": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "sensitivity_recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else None,
            "precision_ppv": float(precision_score(y_true, y_pred, zero_division=0)),
            "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else None,
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
        },
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
    }


@torch.no_grad()
def predict_dataset(df, image_dir, image_col, device, model_dir, n_test):
    ckpts = sorted(Path(model_dir).glob("*.pth"))

    if len(ckpts) == 0:
        raise FileNotFoundError(f"No .pth checkpoints found in {model_dir}")

    print(f"Found {len(ckpts)} checkpoints")

    all_checkpoint_preds = []

    image_index = build_image_index(image_dir)

    image_paths = []
    missing = []

    for v in df[image_col].tolist():
        p = resolve_image_path(v, image_dir, image_index)
        image_paths.append(p)

        if p is None:
            missing.append(v)

    if missing:
        print(f"Warning: {len(missing)} images could not be found.")
        print("First few missing:", missing[:10])

    for i, ckpt_path in enumerate(ckpts, start=1):
        print(f"[{i}/{len(ckpts)}] {ckpt_path.name}")

        kernel_type = siim.extract_kernel_type(ckpt_path)
        config = siim.MODEL_CONFIGS[kernel_type]

        model = siim.build_model(config, device)
        state_dict = siim.load_state_dict_safely(ckpt_path, device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        checkpoint_probs = []

        for image_path in image_paths:
            if image_path is None:
                checkpoint_probs.append(np.nan)
                continue
                
            x = siim.load_image(
                image_path=str(image_path),
                image_size=config["image_size"],
                device=device,
            )

            prob = siim.melanoma_probability_with_tta(
                model=model,
                x=x,
                config=config,
                device=device,
                n_test=n_test,
            )

            checkpoint_probs.append(prob)

        all_checkpoint_preds.append({
            "checkpoint": ckpt_path.name,
            "kernel_type": kernel_type,
            "probs": checkpoint_probs,
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    pred_mat = pd.DataFrame({
        p["checkpoint"]: p["probs"]
        for p in all_checkpoint_preds
    })

    pred_mat.insert(0, "derm7pt_row_id", df.index.values)

    architecture_cols = {}
    for p in all_checkpoint_preds:
        architecture_cols.setdefault(p["kernel_type"], []).append(p["checkpoint"])

    arch_pred_df = pd.DataFrame()
    for arch, cols in architecture_cols.items():
        arch_pred_df[arch] = pred_mat[cols].mean(axis=1)

    pred_mat["siim90_probability_ensemble"] = arch_pred_df.mean(axis=1)

    rank_arch_pred_df = arch_pred_df.rank(pct=True)
    pred_mat["siim90_rank_ensemble"] = rank_arch_pred_df.mean(axis=1)

    return pred_mat


def main():
    parser = argparse.ArgumentParser(
        description="Assess SIIM-ISIC 90-model ensemble on Derm7pt dermoscopy images."
    )

    parser.add_argument("--csv", required=True, help="Derm7pt metadata CSV")
    parser.add_argument("--image_dir", required=True, help="Base folder containing Derm7pt dermoscopy images")
    parser.add_argument("--model_dir", required=True, help="Folder containing SIIM 90 .pth models")
    parser.add_argument("--image_col", required=True, help="Column containing dermoscopy image filename/path")
    parser.add_argument("--diagnosis_col", required=True, help="Column containing Derm7pt diagnosis label")
    parser.add_argument("--output_csv", default="siim90_derm7pt_predictions.csv")
    parser.add_argument("--output_metrics", default="siim90_derm7pt_metrics.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n_test", type=int, default=1, choices=[1, 8])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    args = parser.parse_args()

    device = siim.get_device(args.device)
    print(f"Using device: {device}")

    df = pd.read_csv(args.csv)
    
    print("Available columns:")
    print(df.columns.tolist())

    if args.image_col not in df.columns:
        raise ValueError(f"Image column '{args.image_col}' not found.")

    if args.diagnosis_col not in df.columns:
        raise ValueError(f"Diagnosis column '{args.diagnosis_col}' not found.")
    
    df = df.reset_index(drop=True)
    df["derm7pt_row_id"] = df.index

    labels = df.apply(
        lambda row: get_derm7pt_labels(row, args.diagnosis_col),
        axis=1
    )

    df["y_malignant"] = [x[0] for x in labels]
    df["y_melanoma"] = [x[1] for x in labels]

    print(f"Evaluating {len(df)} Derm7pt dermoscopy images")
    print(f"Melanoma positives: {df['y_melanoma'].sum()}")
    print(f"Malignancy positives: {df['y_malignant'].sum()}")

    pred_mat = predict_dataset(
        df=df,
        image_dir=args.image_dir,
        image_col=args.image_col,
        device=device,
        model_dir=args.model_dir,
        n_test=args.n_test,
    )

    result_df = df.merge(pred_mat, on="derm7pt_row_id", how="left")
    result_df = result_df.dropna(
        subset=["siim90_probability_ensemble", "siim90_rank_ensemble"]
    ).copy()
    result_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")

    score_cols = {
        "probability_ensemble": "siim90_probability_ensemble",
        "rank_ensemble": "siim90_rank_ensemble",
    }

    metrics = {
        "malignancy_prediction": {},
        "melanoma_prediction": {},
    }

    for model_name, score_col in score_cols.items():
        y_prob = result_df[score_col].values

        metrics["malignancy_prediction"][model_name] = {
            "fixed_threshold": compute_metrics(
                result_df["y_malignant"].values,
                y_prob,
                threshold=args.threshold,
                optimize_threshold=False,
            ),
            "optimal_youden_threshold": compute_metrics(
                result_df["y_malignant"].values,
                y_prob,
                threshold=args.threshold,
                optimize_threshold=True,
            ),
        }

        metrics["melanoma_prediction"][model_name] = {
            "fixed_threshold": compute_metrics(
                result_df["y_melanoma"].values,
                y_prob,
                threshold=args.threshold,
                optimize_threshold=False,
            ),
            "optimal_youden_threshold": compute_metrics(
                result_df["y_melanoma"].values,
                y_prob,
                threshold=args.threshold,
                optimize_threshold=True,
            ),
        }

    metrics["notes"] = {
        "dataset": "Derm7pt",
        "model": "SIIM-ISIC 90-model melanoma ensemble",
        "probability_ensemble": (
            "Average melanoma probability across 18 architecture-level predictions."
        ),
        "rank_ensemble": (
            "Official SIIM-style rank ensemble; useful for ROC AUC and PR AUC, "
            "but not calibrated probability."
        ),
        "fixed_threshold": args.threshold,
        "optimal_threshold_method": "Youden index: maximize sensitivity + specificity - 1",
        "n_test": args.n_test,
    }

    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=4))
    print(f"Saved metrics to: {args.output_metrics}")


if __name__ == "__main__":
    main()
