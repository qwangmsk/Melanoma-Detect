#!/usr/bin/env python3
"""
Assess GPT on a derm7pt subset CSV.

Example:
    python gpt_fusion_assess_derm7pt.py \
      --resnet_csv ../milk10k_train_base/resnet_predict_on_derm7pt/resnet_derm7pt_predictions.csv \
      --siim_csv ../SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master/siim90_predict_on_derm7pt/siim90_derm7pt_predictions.csv \
      --model gpt-5.5 \
      --output_csv gpt_fusion_derm7pt_predictions.csv \
      --output_metrics gpt_fusion_derm7pt_metrics.json
        
    python gpt_fusion_assess_derm7pt.py \
      --resnet_csv ../milk10k_train_base/resnet_predict_on_derm7pt/resnet_derm7pt_predictions.csv \
      --siim_csv ../SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master/siim90_predict_on_derm7pt/siim90_derm7pt_predictions.csv \
      --model gpt-5.5 \
      --limit 5        
     
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

FAMILIES = ["MEL", "BCC", "NV", "BKL", "DF", "VASC", "SCCKA", "MISC"]

# Derm7pt Youden thresholds from your model assessments
SIIM_MEL_THRESHOLD = 0.16081724078112586
SIIM_MALIGNANCY_THRESHOLD = 0.14959644144798384

RESNET_MEL_THRESHOLD = 0.18557974882423878
RESNET_MALIGNANCY_THRESHOLD = 0.22138070850633085


def normalize_family(x):
    x = str(x).strip().upper()
    aliases = {
        "MELANOMA": "MEL", "MEL": "MEL",
        "BASAL CELL CARCINOMA": "BCC", "BCC": "BCC",
        "NEVUS": "NV", "NAEVUS": "NV", "NV": "NV",
        "SEBORRHEIC KERATOSIS": "BKL", "SEBORRHOEIC KERATOSIS": "BKL", "SK": "BKL", "BKL": "BKL",
        "DERMATOFIBROMA": "DF", "DF": "DF",
        "VASCULAR": "VASC", "VASC": "VASC",
        "SQUAMOUS CELL CARCINOMA": "SCCKA", "SCC": "SCCKA", "SCCKA": "SCCKA",
        "MISCELLANEOUS": "MISC", "MISC": "MISC",
    }
    return aliases.get(x, "MISC")


def parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


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
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    if y_prob is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))

    return out


def get_col(row, names, default=""):
    for n in names:
        if n in row and pd.notna(row[n]):
            return row[n]
    return default


def build_prompt(row):
    resnet_mel = float(get_col(row, ["resnet_melanoma_probability"], 0))
    resnet_malign = float(get_col(row, ["resnet_clinically_concerning_probability"], 0))
    siim_mel = float(get_col(row, ["siim90_probability_ensemble"], 0))
    siim_rank = float(get_col(row, ["siim90_rank_ensemble"], 0))

    resnet_mel_pos = int(resnet_mel >= RESNET_MEL_THRESHOLD)
    resnet_malign_pos = int(resnet_malign >= RESNET_MALIGNANCY_THRESHOLD)
    siim_mel_pos = int(siim_mel >= SIIM_MEL_THRESHOLD)

    return f"""
You are a diagnostic fusion agent for a research classification task.

You are NOT allowed to use images. Use ONLY the model-output evidence below.

Your role:
- Do NOT independently invent a diagnosis.
- Integrate evidence from two AI systems.
- Treat the ResNet as the broader differential-diagnosis model.
- Treat the SIIM ensemble as a melanoma-specialist model.
- For melanoma prediction, trust SIIM more than ResNet because SIIM was specifically optimized for melanoma detection.
- For broad malignancy prediction, use SIIM mainly as melanoma evidence; use ResNet to assess non-melanoma malignancy such as BCC or SCCKA.
- A low SIIM score does NOT rule out non-melanoma malignancy.

Family labels must be exactly one of:
MEL, BCC, NV, BKL, DF, VASC, SCCKA, MISC

ResNet outputs:
- Top1 diagnosis: {get_col(row, ["top1_diagnosis", "resnet_top1_diagnosis"])}
- Top1 family: {get_col(row, ["top1_family", "resnet_top1_family"])}
- Top1 probability: {get_col(row, ["top1_probability", "resnet_top1_probability"])}
- Top2 diagnosis: {get_col(row, ["top2_diagnosis", "resnet_top2_diagnosis"])}
- Top2 family: {get_col(row, ["top2_family", "resnet_top2_family"])}
- Top2 probability: {get_col(row, ["top2_probability", "resnet_top2_probability"])}
- Top3 diagnosis: {get_col(row, ["top3_diagnosis", "resnet_top3_diagnosis"])}
- Top3 family: {get_col(row, ["top3_family", "resnet_top3_family"])}
- Top3 probability: {get_col(row, ["top3_probability", "resnet_top3_probability"])}

ResNet family probabilities:
- MEL: {get_col(row, ["family_prob_MEL"], 0)}
- BCC: {get_col(row, ["family_prob_BCC"], 0)}
- NV: {get_col(row, ["family_prob_NV"], 0)}
- BKL: {get_col(row, ["family_prob_BKL"], 0)}
- DF: {get_col(row, ["family_prob_DF"], 0)}
- VASC: {get_col(row, ["family_prob_VASC"], 0)}
- SCCKA: {get_col(row, ["family_prob_SCCKA"], 0)}
- AKIEC: {get_col(row, ["family_prob_AKIEC"], 0)}
- MAL_OTH: {get_col(row, ["family_prob_MAL_OTH"], 0)}
- BEN_OTH: {get_col(row, ["family_prob_BEN_OTH"], 0)}
- INF: {get_col(row, ["family_prob_INF"], 0)}

ResNet threshold-adjusted interpretation:
- ResNet melanoma probability: {resnet_mel}
- ResNet melanoma threshold: {RESNET_MEL_THRESHOLD}
- ResNet melanoma positive: {resnet_mel_pos}
- ResNet clinically concerning probability: {resnet_malign}
- ResNet malignancy threshold: {RESNET_MALIGNANCY_THRESHOLD}
- ResNet malignancy positive: {resnet_malign_pos}

SIIM melanoma-specialist ensemble:
- SIIM melanoma probability ensemble: {siim_mel}
- SIIM melanoma rank ensemble: {siim_rank}
- SIIM melanoma threshold: {SIIM_MEL_THRESHOLD}
- SIIM melanoma positive: {siim_mel_pos}

Suggested weighting:
- Melanoma prediction: SIIM 70-80%, ResNet 20-30%.
- Broad malignancy prediction: SIIM contributes strong melanoma evidence only; ResNet contributes broader non-melanoma malignancy evidence.

Return ONLY valid JSON:

{{
  "primary_diagnosis": "",
  "primary_family": "",
  "top1_probability": 0.0,
  "top3_differential": [
    {{"diagnosis": "", "family": "", "probability": 0.0}},
    {{"diagnosis": "", "family": "", "probability": 0.0}},
    {{"diagnosis": "", "family": "", "probability": 0.0}}
  ],
  "melanoma_probability": 0.0,
  "malignancy_probability": 0.0,
  "melanoma_prediction": 0,
  "malignancy_prediction": 0,
  "followed_resnet": true,
  "followed_siim": true,
  "resnet_weight": 0.0,
  "siim_weight": 0.0,
  "confidence": 0.0,
  "melanoma_decision_basis": "",
  "malignancy_decision_basis": "",
  "brief_rationale": ""
}}
"""


def ask_llm(client, model, row):
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": build_prompt(row)}],
            }
        ],
    )
    return parse_json_response(response.output_text)


def pad_top3(top3):
    if not isinstance(top3, list):
        top3 = []
    while len(top3) < 3:
        top3.append({"diagnosis": "", "family": "MISC", "probability": 0.0})
    return top3[:3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet_csv", required=True)
    parser.add_argument("--siim_csv", required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--output_csv", default="gpt_fusion_derm7pt_predictions.csv")
    parser.add_argument("--output_metrics", default="gpt_fusion_derm7pt_metrics.json")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    client = OpenAI()

    resnet = pd.read_csv(args.resnet_csv)
    siim = pd.read_csv(args.siim_csv)

    siim_small = siim[
        ["derm7pt_row_id", "siim90_probability_ensemble", "siim90_rank_ensemble"]
    ].copy()

    merged = resnet.merge(
        siim_small,
        left_on="row_id",
        right_on="derm7pt_row_id",
        how="left",
    )

    merged = merged.dropna(
        subset=[
            "siim90_probability_ensemble",
            "siim90_rank_ensemble",
            "resnet_melanoma_probability",
            "resnet_clinically_concerning_probability",
        ]
    ).copy()

    if args.limit:
        merged = merged.head(args.limit).copy()

    results = []

    for _, row in merged.iterrows():
        print(f"[{len(results)+1}/{len(merged)}] row_id={row['row_id']} diagnosis={row['diagnosis']}")

        try:
            pred = ask_llm(client, args.model, row)
        except Exception as e:
            print(f"Error at row {row['row_id']}: {e}")
            continue

        top3 = pad_top3(pred.get("top3_differential", []))

        gt_family = normalize_family(row.get("gt_family", "MISC"))

        top1_family = normalize_family(pred.get("primary_family", top3[0].get("family", "MISC")))
        top2_family = normalize_family(top3[1].get("family", "MISC"))
        top3_family = normalize_family(top3[2].get("family", "MISC"))

        resnet_top1_family = normalize_family(get_col(row, ["top1_family", "resnet_top1_family"], "MISC"))

        resnet_mel_prob = float(row.get("resnet_melanoma_probability", 0))
        resnet_malign_prob = float(row.get("resnet_clinically_concerning_probability", 0))
        siim_mel_prob = float(row.get("siim90_probability_ensemble", 0))

        resnet_melanoma_pred = int(resnet_mel_prob >= RESNET_MEL_THRESHOLD)
        resnet_malignancy_pred = int(resnet_malign_prob >= RESNET_MALIGNANCY_THRESHOLD)
        siim_melanoma_pred = int(siim_mel_prob >= SIIM_MEL_THRESHOLD)

        fusion_melanoma_pred = int(pred.get("melanoma_prediction", 0))
        fusion_malignancy_pred = int(pred.get("malignancy_prediction", 0))

        out = {
            "row_id": row["row_id"],
            "case_num": row.get("case_num", ""),
            "clinic": row.get("clinic", ""),
            "derm": row.get("derm", ""),
            "diagnosis": row.get("diagnosis", ""),
            "gt_family": gt_family,
            "y_malignant": int(row["y_malignant"]),
            "y_melanoma": int(row["y_melanoma"]),

            "resnet_top1_family": resnet_top1_family,
            "resnet_melanoma_probability": resnet_mel_prob,
            "resnet_clinically_concerning_probability": resnet_malign_prob,
            "resnet_melanoma_threshold": RESNET_MEL_THRESHOLD,
            "resnet_malignancy_threshold": RESNET_MALIGNANCY_THRESHOLD,
            "resnet_melanoma_prediction_youden": resnet_melanoma_pred,
            "resnet_malignancy_prediction_youden": resnet_malignancy_pred,

            "siim90_probability_ensemble": siim_mel_prob,
            "siim90_rank_ensemble": row.get("siim90_rank_ensemble", np.nan),
            "siim90_melanoma_threshold": SIIM_MEL_THRESHOLD,
            "siim90_melanoma_prediction_youden": siim_melanoma_pred,

            "resnet_siim_melanoma_agree_youden": int(resnet_melanoma_pred == siim_melanoma_pred),

            "gpt_fusion_top1_diagnosis": pred.get("primary_diagnosis", ""),
            "gpt_fusion_top1_family": top1_family,
            "gpt_fusion_top1_probability": float(pred.get("top1_probability", np.nan)),

            "gpt_fusion_top2_diagnosis": top3[1].get("diagnosis", ""),
            "gpt_fusion_top2_family": top2_family,
            "gpt_fusion_top2_probability": float(top3[1].get("probability", 0.0)),

            "gpt_fusion_top3_diagnosis": top3[2].get("diagnosis", ""),
            "gpt_fusion_top3_family": top3_family,
            "gpt_fusion_top3_probability": float(top3[2].get("probability", 0.0)),

            "gpt_fusion_melanoma_probability": float(pred.get("melanoma_probability", np.nan)),
            "gpt_fusion_malignancy_probability": float(pred.get("malignancy_probability", np.nan)),
            "gpt_fusion_melanoma_prediction": fusion_melanoma_pred,
            "gpt_fusion_malignancy_prediction": fusion_malignancy_pred,

            "followed_resnet": bool(pred.get("followed_resnet", False)),
            "followed_siim": bool(pred.get("followed_siim", False)),
            "resnet_weight": float(pred.get("resnet_weight", np.nan)),
            "siim_weight": float(pred.get("siim_weight", np.nan)),
            "fusion_confidence": float(pred.get("confidence", np.nan)),
            "melanoma_decision_basis": pred.get("melanoma_decision_basis", ""),
            "malignancy_decision_basis": pred.get("malignancy_decision_basis", ""),
            "gpt_fusion_rationale": pred.get("brief_rationale", ""),
            "gpt_fusion_raw_top3": json.dumps(top3),
        }

        out["gpt_fusion_top1_family_correct"] = int(out["gpt_fusion_top1_family"] == gt_family)
        out["gpt_fusion_top3_family_correct"] = int(
            gt_family in [
                out["gpt_fusion_top1_family"],
                out["gpt_fusion_top2_family"],
                out["gpt_fusion_top3_family"],
            ]
        )

        out["fusion_changed_family_from_resnet"] = int(out["gpt_fusion_top1_family"] != resnet_top1_family)
        out["fusion_changed_melanoma_from_resnet"] = int(fusion_melanoma_pred != resnet_melanoma_pred)
        out["fusion_changed_malignancy_from_resnet"] = int(fusion_malignancy_pred != resnet_malignancy_pred)
        out["fusion_changed_melanoma_from_siim"] = int(fusion_melanoma_pred != siim_melanoma_pred)

        results.append(out)
        time.sleep(args.sleep)

    result_df = pd.DataFrame(results)

    result_df = result_df.dropna(
        subset=[
            "gpt_fusion_melanoma_probability",
            "gpt_fusion_malignancy_probability",
        ]
    ).copy()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(args.output_csv, index=False)

    metrics = {
        "family_diagnosis": {
            "top1_family_accuracy": float(result_df["gpt_fusion_top1_family_correct"].mean()),
            "top3_family_accuracy": float(result_df["gpt_fusion_top3_family_correct"].mean()),
            "n": int(len(result_df)),
        },
        "malignancy_prediction": compute_binary_metrics(
            result_df["y_malignant"].values,
            result_df["gpt_fusion_malignancy_prediction"].values,
            result_df["gpt_fusion_malignancy_probability"].values,
        ),
        "melanoma_prediction": compute_binary_metrics(
            result_df["y_melanoma"].values,
            result_df["gpt_fusion_melanoma_prediction"].values,
            result_df["gpt_fusion_melanoma_probability"].values,
        ),
        "fusion_behavior": {
            "changed_family_from_resnet": float(result_df["fusion_changed_family_from_resnet"].mean()),
            "changed_melanoma_from_resnet": float(result_df["fusion_changed_melanoma_from_resnet"].mean()),
            "changed_malignancy_from_resnet": float(result_df["fusion_changed_malignancy_from_resnet"].mean()),
            "changed_melanoma_from_siim": float(result_df["fusion_changed_melanoma_from_siim"].mean()),
            "resnet_siim_melanoma_agreement_youden": float(result_df["resnet_siim_melanoma_agree_youden"].mean()),
            "mean_resnet_weight": float(result_df["resnet_weight"].mean()),
            "mean_siim_weight": float(result_df["siim_weight"].mean()),
            "mean_fusion_confidence": float(result_df["fusion_confidence"].mean()),
        },
        "notes": {
            "dataset": "Derm7pt",
            "model": args.model,
            "input_to_llm": "Only ResNet and SIIM prediction outputs; no images.",
            "thresholds": {
                "SIIM_MEL_THRESHOLD": SIIM_MEL_THRESHOLD,
                "RESNET_MEL_THRESHOLD": RESNET_MEL_THRESHOLD,
                "RESNET_MALIGNANCY_THRESHOLD": RESNET_MALIGNANCY_THRESHOLD,
            },
            "purpose": "Evaluate whether LLM fusion of threshold-adjusted model outputs improves over image-only LLM assessment.",
        },
    }

    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=4))
    print(f"Saved predictions to: {args.output_csv}")
    print(f"Saved metrics to: {args.output_metrics}")


if __name__ == "__main__":
    main()
