#!/usr/bin/env python3

"""
Assess GPT on a derm7pt subset CSV.

Example:
    python gpt_assess_derm7pt.py \
      --csv ../derm7pt/release_v0/meta/meta.csv \
      --image_dir ../derm7pt/release_v0/images \
      --model gpt-5.5 \
      --output_csv gpt_derm7pt_predictions.csv \
      --output_metrics gpt_derm7pt_metrics.json 
      
    python gpt_assess_derm7pt.py \
      --csv ../derm7pt/release_v0/meta/meta.csv \
      --image_dir ../derm7pt/release_v0/images \
      --model gpt-5.5 \
      --limit 5      
"""

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def encode_image(path):
    path = Path(path)
    ext = path.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:image/{ext};base64,{b64}"


def derm7pt_labels(dx):
    dx = str(dx).strip().lower()

    y_melanoma = int("melanoma" in dx)

    malignant_terms = [
        "melanoma",
        "basal cell carcinoma",
        "squamous cell carcinoma",
    ]
    y_malignant = int(any(x in dx for x in malignant_terms))

    return y_malignant, y_melanoma


def derm7pt_family(dx):
    dx = str(dx).strip().lower()

    if "melanoma" in dx:
        return "MEL"
    if "basal cell carcinoma" in dx:
        return "BCC"
    if "nevus" in dx or "naevus" in dx:
        return "NV"
    if "seborrheic keratosis" in dx or "seborrhoeic keratosis" in dx:
        return "BKL"
    if "dermatofibroma" in dx:
        return "DF"
    if "vascular" in dx or "angioma" in dx or "hemangioma" in dx:
        return "VASC"
    if "squamous cell carcinoma" in dx:
        return "SCCKA"

    return "MISC"


def normalize_family(fam):
    fam = str(fam).strip().upper()

    aliases = {
        "MELANOMA": "MEL",
        "MEL": "MEL",
        "BCC": "BCC",
        "BASAL CELL CARCINOMA": "BCC",
        "NEVUS": "NV",
        "NAEVUS": "NV",
        "NV": "NV",
        "SEBORRHEIC KERATOSIS": "BKL",
        "SEBORRHOEIC KERATOSIS": "BKL",
        "SK": "BKL",
        "BKL": "BKL",
        "DF": "DF",
        "DERMATOFIBROMA": "DF",
        "VASC": "VASC",
        "VASCULAR": "VASC",
        "SCC": "SCCKA",
        "SCCKA": "SCCKA",
        "SQUAMOUS CELL CARCINOMA": "SCCKA",
        "MISC": "MISC",
        "MISCELLANEOUS": "MISC",
    }

    return aliases.get(fam, "MISC")


def parse_json_response(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


def ask_model(client, model, clinic_path, derm_path):
    clinic_b64 = encode_image(clinic_path)
    derm_b64 = encode_image(derm_path)

    prompt = """
You are assessing paired dermatology images: one clinical image and one dermoscopic image.

This is a research classification task. Do not provide medical advice to a patient.

Return ONLY valid JSON with exactly this structure:

{
  "primary_diagnosis": "",
  "primary_family": "",
  "top1_probability": 0.0,
  "top3_differential": [
    {
      "diagnosis": "",
      "family": "",
      "probability": 0.0
    },
    {
      "diagnosis": "",
      "family": "",
      "probability": 0.0
    },
    {
      "diagnosis": "",
      "family": "",
      "probability": 0.0
    }
  ],
  "melanoma_probability": 0.0,
  "malignancy_probability": 0.0,
  "melanoma_prediction": 0,
  "malignancy_prediction": 0,
  "brief_rationale": ""
}

Definitions:
- melanoma_probability: probability that the lesion is melanoma.
- malignancy_probability: probability that the lesion is malignant, including melanoma, basal cell carcinoma, squamous cell carcinoma, or other malignant skin tumors.
- melanoma_prediction: 1 if melanoma is likely, otherwise 0.
- malignancy_prediction: 1 if any malignancy is likely, otherwise 0.
- Use probabilities from 0 to 1.

The family field MUST be exactly one of:
MEL, BCC, NV, BKL, DF, VASC, SCCKA, MISC

Family definitions:
- MEL: melanoma
- BCC: basal cell carcinoma
- NV: nevus
- BKL: seborrheic keratosis / benign keratosis
- DF: dermatofibroma
- VASC: vascular lesion
- SCCKA: squamous cell carcinoma / keratoacanthoma
- MISC: other or uncertain diagnosis
"""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_text", "text": "Clinical image:"},
                    {"type": "input_image", "image_url": clinic_b64},
                    {"type": "input_text", "text": "Dermoscopic image:"},
                    {"type": "input_image", "image_url": derm_b64},
                ],
            }
        ],
    )

    return parse_json_response(response.output_text)


def compute_binary_metrics(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "precision_ppv": float(precision_score(y_true, y_pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    if y_prob is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))

    return out


def pad_top3(top3):
    if not isinstance(top3, list):
        top3 = []

    while len(top3) < 3:
        top3.append(
            {
                "diagnosis": "",
                "family": "MISC",
                "probability": 0.0,
            }
        )

    return top3[:3]


def main():
    parser = argparse.ArgumentParser(
        description="Assess ChatGPT/OpenAI vision model on Derm7pt paired images."
    )

    parser.add_argument("--csv", required=True, help="Derm7pt meta.csv")
    parser.add_argument("--image_dir", required=True, help="Derm7pt image base directory")
    parser.add_argument("--model", default="gpt-5.5", help="OpenAI vision-capable model")
    parser.add_argument("--clinic_col", default="clinic")
    parser.add_argument("--derm_col", default="derm")
    parser.add_argument("--diagnosis_col", default="diagnosis")
    parser.add_argument("--output_csv", default="gpt_derm7pt_predictions.csv")
    parser.add_argument("--output_metrics", default="gpt_derm7pt_metrics.json")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    client = OpenAI()
    image_dir = Path(args.image_dir)

    df = pd.read_csv(args.csv)

    for col in [args.clinic_col, args.derm_col, args.diagnosis_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available columns: {df.columns.tolist()}")

    if args.limit:
        df = df.head(args.limit).copy()

    results = []

    for i, row in df.iterrows():
        clinic_path = image_dir / str(row[args.clinic_col])
        derm_path = image_dir / str(row[args.derm_col])

        if not clinic_path.exists() or not derm_path.exists():
            print(f"Skipping missing images at row {i}: {clinic_path}, {derm_path}")
            continue

        y_malignant, y_melanoma = derm7pt_labels(row[args.diagnosis_col])
        gt_family = derm7pt_family(row[args.diagnosis_col])

        print(f"[{len(results)+1}/{len(df)}] row={i}, diagnosis={row[args.diagnosis_col]}")

        try:
            pred = ask_model(
                client=client,
                model=args.model,
                clinic_path=clinic_path,
                derm_path=derm_path,
            )
        except Exception as e:
            print(f"Error at row {i}: {e}")
            continue

        top3 = pad_top3(pred.get("top3_differential", []))

        top1_family = normalize_family(pred.get("primary_family", top3[0].get("family", "MISC")))
        top2_family = normalize_family(top3[1].get("family", "MISC"))
        top3_family = normalize_family(top3[2].get("family", "MISC"))

        top1_diagnosis = pred.get("primary_diagnosis", top3[0].get("diagnosis", ""))

        try:
            top1_probability = float(pred.get("top1_probability", top3[0].get("probability", np.nan)))
        except Exception:
            top1_probability = np.nan

        try:
            melanoma_probability = float(pred.get("melanoma_probability", np.nan))
        except Exception:
            melanoma_probability = np.nan

        try:
            malignancy_probability = float(pred.get("malignancy_probability", np.nan))
        except Exception:
            malignancy_probability = np.nan

        out = {
            "row_id": i,
            "case_num": row.get("case_num", ""),
            "clinic": row[args.clinic_col],
            "derm": row[args.derm_col],
            "diagnosis": row[args.diagnosis_col],
            "gt_family": gt_family,
            "y_malignant": y_malignant,
            "y_melanoma": y_melanoma,

            "gpt_top1_diagnosis": top1_diagnosis,
            "gpt_top1_family": top1_family,
            "gpt_top1_probability": top1_probability,

            "gpt_top2_diagnosis": top3[1].get("diagnosis", ""),
            "gpt_top2_family": top2_family,
            "gpt_top2_probability": float(top3[1].get("probability", 0.0)),

            "gpt_top3_diagnosis": top3[2].get("diagnosis", ""),
            "gpt_top3_family": top3_family,
            "gpt_top3_probability": float(top3[2].get("probability", 0.0)),

            "gpt_top3_differential_raw": json.dumps(top3),

            "gpt_melanoma_probability": melanoma_probability,
            "gpt_malignancy_probability": malignancy_probability,
            "gpt_melanoma_prediction": int(pred.get("melanoma_prediction", 0)),
            "gpt_malignancy_prediction": int(pred.get("malignancy_prediction", 0)),
            "gpt_brief_rationale": pred.get("brief_rationale", ""),
        }

        out["gpt_top1_family_correct"] = int(out["gpt_top1_family"] == gt_family)
        out["gpt_top3_family_correct"] = int(
            gt_family in [
                out["gpt_top1_family"],
                out["gpt_top2_family"],
                out["gpt_top3_family"],
            ]
        )

        results.append(out)
        time.sleep(args.sleep)

    result_df = pd.DataFrame(results)

    result_df = result_df.dropna(
        subset=[
            "gpt_melanoma_probability",
            "gpt_malignancy_probability",
            "gpt_melanoma_prediction",
            "gpt_malignancy_prediction",
        ]
    ).copy()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")

    metrics = {
        "family_diagnosis": {
            "top1_family_accuracy": float(result_df["gpt_top1_family_correct"].mean()),
            "top3_family_accuracy": float(result_df["gpt_top3_family_correct"].mean()),
            "n": int(len(result_df)),
        },
        "malignancy_prediction": compute_binary_metrics(
            y_true=result_df["y_malignant"].values,
            y_pred=result_df["gpt_malignancy_prediction"].values,
            y_prob=result_df["gpt_malignancy_probability"].values,
        ),
        "melanoma_prediction": compute_binary_metrics(
            y_true=result_df["y_melanoma"].values,
            y_pred=result_df["gpt_melanoma_prediction"].values,
            y_prob=result_df["gpt_melanoma_probability"].values,
        ),
        "notes": {
            "dataset": "Derm7pt",
            "model": args.model,
            "inputs": "clinical image + dermoscopic image",
            "family_labels": ["MEL", "BCC", "NV", "BKL", "DF", "VASC", "SCCKA", "MISC"],
            "ground_truth_malignant": "melanoma, basal cell carcinoma, squamous cell carcinoma",
            "ground_truth_melanoma": "diagnosis contains melanoma",
        },
    }

    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=4))
    print(f"Saved metrics to: {args.output_metrics}")


if __name__ == "__main__":
    main()
