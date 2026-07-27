#!/usr/bin/env python3

"""
Assess ResNet50 on a derm7pt subset CSV.

Example:
    python resnet_assess_derm7pt.py \
      --csv ../derm7pt/release_v0/meta/meta.csv \
      --image_dir ../derm7pt/release_v0/images \
      --run_dir runs/20260605_083158 \
      --topk 5 \
      --output_csv resnet_predict_on_derm7pt/resnet_derm7pt_predictions.csv \
      --output_metrics resnet_predict_on_derm7pt/resnet_derm7pt_metrics.json \
      --topk 3 
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve
)

from utils.data import image_transforms
from utils.model import MultimodalNetSimple


def get_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_image(path, transform, device):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def load_model(checkpoint_path, device, num_classes=48):
    model = MultimodalNetSimple(num_classes=num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_models(run_dir, device):
    models = []
    for fold in range(5):
        ckpt = Path(run_dir) / f"model_fold_{fold}_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        models.append(load_model(ckpt, device))
    return models


@torch.no_grad()
def predict_pair(models, clinic_path, derm_path, transform, device):
    clinic = load_image(clinic_path, transform, device)
    derm = load_image(derm_path, transform, device)

    fold_probs = []
    for model in models:
        logits = model(clinic, derm)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        fold_probs.append(probs)

    fold_probs = torch.stack(fold_probs)
    mean_probs = fold_probs.mean(dim=0)
    sd_probs = fold_probs.std(dim=0)

    return mean_probs, sd_probs


def derm7pt_family(dx):
    dx = str(dx).strip().lower()

    if "melanoma" in dx:
        return "MEL"
    if "basal cell carcinoma" in dx or dx == "bcc":
        return "BCC"
    if "nevus" in dx or "naevus" in dx:
        return "NV"
    if "seborrheic keratosis" in dx or "seborrhoeic keratosis" in dx:
        return "BKL"
    if "dermatofibroma" in dx:
        return "DF"
    if "vascular" in dx or "angioma" in dx or "hemangioma" in dx:
        return "VASC"
    if "squamous cell carcinoma" in dx or dx == "scc":
        return "SCCKA"

    return "MISC"


def get_binary_labels(gt_family):
    y_melanoma = int(gt_family == "MEL")
    y_malignant = int(gt_family in {"MEL", "BCC", "SCCKA", "MAL_OTH"})
    return y_malignant, y_melanoma


def family_probabilities(mean_probs, idx_to_class, label_to_simplified):
    fam_probs = {}
    for i, p in enumerate(mean_probs.tolist()):
        dx = idx_to_class[str(i)]
        fam = label_to_simplified.get(dx, "NA")
        fam_probs[fam] = fam_probs.get(fam, 0.0) + p
    return fam_probs


def compute_metrics(y_true, y_prob, threshold=0.5, optimize_threshold=False):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return None

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
        },
    }


def unique_top_families(mean_probs, idx_to_class, label_to_simplified, topk=3):
    sorted_idx = torch.argsort(mean_probs, descending=True).tolist()

    families = []
    diagnoses = []

    for idx in sorted_idx:
        dx = idx_to_class[str(idx)]
        fam = label_to_simplified.get(dx, "NA")
        prob = float(mean_probs[idx].item())

        if fam not in families:
            families.append(fam)
            diagnoses.append((dx, fam, prob))

        if len(families) >= topk:
            break

    return diagnoses


def resolve_path(base_dir, value):
    value = str(value).strip()
    p = Path(value)

    if p.is_absolute():
        return p

    return Path(base_dir) / p


def main():
    parser = argparse.ArgumentParser(
        description="Assess MILK10K multimodal ResNet ensemble on Derm7pt."
    )

    parser.add_argument("--csv", required=True, help="Derm7pt meta.csv")
    parser.add_argument("--image_dir", required=True, help="Base image folder containing Derm7pt subfolders")
    parser.add_argument("--run_dir", required=True, help="MILK10K run directory containing checkpoints and JSON files")
    parser.add_argument("--clinic_col", default="clinic")
    parser.add_argument("--derm_col", default="derm")
    parser.add_argument("--diagnosis_col", default="diagnosis")
    parser.add_argument("--output_csv", default="resnet_derm7pt_predictions.csv")
    parser.add_argument("--output_metrics", default="resnet_derm7pt_metrics.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    df = pd.read_csv(args.csv)
    print("Available columns:", df.columns.tolist())

    for col in [args.clinic_col, args.derm_col, args.diagnosis_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in metadata.")

    run_dir = Path(args.run_dir)

    with open(run_dir / "dataset-idx_to_class.json") as f:
        idx_to_class = json.load(f)

    with open(run_dir / "dataset-label_to_simplified.json") as f:
        label_to_simplified = json.load(f)

    transform = image_transforms["val"]
    models = load_models(run_dir, device)

    results = []

    malignant_families = {"MEL", "BCC", "SCCKA", "MAL_OTH"}
    premalignant_families = {"AKIEC"}
    benign_families = {"NV", "BKL", "DF", "INF", "VASC", "BEN_OTH"}

    for i, row in df.iterrows():
        clinic_path = resolve_path(args.image_dir, row[args.clinic_col])
        derm_path = resolve_path(args.image_dir, row[args.derm_col])

        if not clinic_path.exists() or not derm_path.exists():
            print(f"Skipping missing pair at row {i}: {clinic_path}, {derm_path}")
            continue

        print(f"[{len(results)+1}/{len(df)}] row={i}")

        mean_probs, sd_probs = predict_pair(
            models=models,
            clinic_path=clinic_path,
            derm_path=derm_path,
            transform=transform,
            device=device,
        )

        fam_probs = family_probabilities(mean_probs, idx_to_class, label_to_simplified)

        gt_family = derm7pt_family(row[args.diagnosis_col])
        y_malignant, y_melanoma = get_binary_labels(gt_family)

        melanoma_prob = fam_probs.get("MEL", 0.0)
        invasive_malignancy_prob = sum(fam_probs.get(f, 0.0) for f in malignant_families)
        premalignant_prob = sum(fam_probs.get(f, 0.0) for f in premalignant_families)
        clinically_concerning_prob = invasive_malignancy_prob + premalignant_prob
        benign_prob = sum(fam_probs.get(f, 0.0) for f in benign_families)

        top_fams = unique_top_families(
            mean_probs,
            idx_to_class,
            label_to_simplified,
            topk=args.topk,
        )

        top1_family = top_fams[0][1] if len(top_fams) > 0 else "NA"
        topk_family_list = [x[1] for x in top_fams]

        out = {
            "row_id": i,
            "case_num": row.get("case_num", ""),
            "clinic": row[args.clinic_col],
            "derm": row[args.derm_col],
            "diagnosis": row[args.diagnosis_col],
            "gt_family": gt_family,
            "y_malignant": y_malignant,
            "y_melanoma": y_melanoma,
            "resnet_melanoma_probability": float(melanoma_prob),
            "resnet_invasive_malignancy_probability": float(invasive_malignancy_prob),
            "resnet_premalignant_AKIEC_probability": float(premalignant_prob),
            "resnet_clinically_concerning_probability": float(clinically_concerning_prob),
            "resnet_benign_probability": float(benign_prob),
            "resnet_top1_family": top1_family,
            "resnet_top1_family_correct": int(top1_family == gt_family),
            "resnet_topk_family_correct": int(gt_family in topk_family_list),
        }

        for fam, p in fam_probs.items():
            out[f"family_prob_{fam}"] = float(p)

        for rank, (dx, fam, prob) in enumerate(top_fams, start=1):
            out[f"top{rank}_diagnosis"] = dx
            out[f"top{rank}_family"] = fam
            out[f"top{rank}_probability"] = prob

        results.append(out)

    result_df = pd.DataFrame(results)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")

    y_malignant = result_df["y_malignant"].values
    y_melanoma = result_df["y_melanoma"].values

    metrics = {
        "family_diagnosis": {
            "top1_family_accuracy": float(result_df["resnet_top1_family_correct"].mean()),
            f"top{args.topk}_family_accuracy": float(result_df["resnet_topk_family_correct"].mean()),
            "n": int(len(result_df)),
        },
        "malignancy_prediction": {
            "invasive_malignancy_probability": {
                "fixed_threshold": compute_metrics(
                    y_malignant,
                    result_df["resnet_invasive_malignancy_probability"].values,
                    threshold=args.threshold,
                    optimize_threshold=False,
                ),
                "optimal_youden_threshold": compute_metrics(
                    y_malignant,
                    result_df["resnet_invasive_malignancy_probability"].values,
                    threshold=args.threshold,
                    optimize_threshold=True,
                ),
            },
            "clinically_concerning_probability": {
                "fixed_threshold": compute_metrics(
                    y_malignant,
                    result_df["resnet_clinically_concerning_probability"].values,
                    threshold=args.threshold,
                    optimize_threshold=False,
                ),
                "optimal_youden_threshold": compute_metrics(
                    y_malignant,
                    result_df["resnet_clinically_concerning_probability"].values,
                    threshold=args.threshold,
                    optimize_threshold=True,
                ),
            },
        },
        "melanoma_prediction": {
            "melanoma_probability": {
                "fixed_threshold": compute_metrics(
                    y_melanoma,
                    result_df["resnet_melanoma_probability"].values,
                    threshold=args.threshold,
                    optimize_threshold=False,
                ),
                "optimal_youden_threshold": compute_metrics(
                    y_melanoma,
                    result_df["resnet_melanoma_probability"].values,
                    threshold=args.threshold,
                    optimize_threshold=True,
                ),
            }
        },
        "notes": {
            "model": "MILK10K 5-fold multimodal ResNet ensemble",
            "dataset": "Derm7pt",
            "inputs": "clinic + dermoscopic image",
            "family_mapping": {
                "Derm7pt melanoma": "MEL",
                "Derm7pt basal cell carcinoma": "BCC",
                "Derm7pt nevus": "NV",
                "Derm7pt seborrheic keratosis": "BKL",
                "Derm7pt miscellaneous": "MISC unless mapped by diagnosis string",
            },
            "malignancy_probability": "MEL + BCC + SCCKA + MAL_OTH",
            "clinically_concerning_probability": "MEL + BCC + SCCKA + MAL_OTH + AKIEC",
            "AKIEC_note": "AKIEC is reported separately as premalignant/in situ and included only in clinically_concerning_probability.",
            "optimal_threshold_method": "Youden index: maximize sensitivity + specificity - 1",
        },
    }

    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=4))
    print(f"Saved metrics to: {args.output_metrics}")


if __name__ == "__main__":
    main()
