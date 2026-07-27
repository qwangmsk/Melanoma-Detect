#!/usr/bin/env python3

"""
Evaluating GPT-5.2 for malignancy discrimination on MILK10k

How to run:
  python milk10k_malignancy_eval.py
"""

import os, json, time, base64
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

from openai import OpenAI

# =========================
# CONFIG
# =========================
CSV_PATH = "../script-preprocess/milk10k_500.csv"

# The folder containing all jpg images (derm + clinical)
IMAGES_ROOT = Path(r"../milk10k/images")

MODEL = "gpt-5.2"
SLEEP_S = 0.25
MAX_RETRIES = 3

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# =========================
# Image utilities
# =========================
def build_isic_index(images_root: Path):
    """
    Build a mapping: isic_id -> image filepath by scanning jpg/jpeg/png recursively.
    Assume common pattern of ISIC filenames: "ISIC_1234567"
    """
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in images_root.rglob("*") if p.suffix.lower() in exts]

    index = {}
    for p in files:
        name = p.stem
        index[name] = p
        if name.startswith("ISIC_"):
            index[name.replace("ISIC_", "")] = p
 
    return index, files


def find_image_path(isic_id: str, index: dict, all_files: list):
    """
    Find image path for isic_id using:
    1) direct key match
    2) ISIC_<id> match
    3) substring match over all files (fallback)
    """
    if isic_id in index:
        return index[isic_id]

    key2 = f"ISIC_{isic_id}"
    if key2 in index:
        return index[key2]

    # fallback substring search
    hits = [p for p in all_files if isic_id in p.stem]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # pick shortest name
        hits.sort(key=lambda p: len(p.stem))
        return hits[0]

    raise FileNotFoundError(f"Could not find an image file for isic_id={isic_id} under {IMAGES_ROOT}")


def encode_image_to_data_url(path: Path) -> str:
    b = path.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    # best-effort mime
    mime = "image/jpeg" if path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64}"


# =========================
# GPT calling
# =========================
def build_prompt():
    return (
        "Task: classify the lesion as Malignant or Benign based on this dermoscopic image.\n"
        "Return ONLY valid JSON with keys:\n"
        "  pred: 'Malignant' or 'Benign'\n"
        "  confidence: number from 0 to 1\n"
        "No extra keys. No prose."
    )

def build_prompt2():
    return (
        "Task: classify the lesion as Malignant or Benign based on this dermoscopic image and the clinical close-up.\n"
        "Return ONLY valid JSON with keys:\n"
        "  pred: 'Malignant' or 'Benign'\n"
        "  confidence: number from 0 to 1\n"
        "No extra keys. No prose."
    )

def call_gpt_with_images(image_data_urls, prompt: str) -> dict:
    content = [{"type": "input_text", "text": prompt}]
    for url in image_data_urls:
        content.append({"type": "input_image", "image_url": url})

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": content}],
            )
            text = resp.output_text.strip()
            return json.loads(text)
        except Exception as e:
            last_err = e
            time.sleep(0.8 * attempt)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_err}")


# =========================
# Data prep
# =========================
def lesion_level_table(df_pairs: pd.DataFrame) -> pd.DataFrame:
    tmp = df_pairs.copy()
    tmp["image_type_norm"] = tmp["image_type"].astype(str).str.lower()

    derm = tmp[tmp["image_type_norm"].str.contains("derm")].copy()
    clin = tmp[tmp["image_type_norm"].str.contains("clinic")].copy()

    merged = derm.merge(clin, on="lesion_id", suffixes=("_derm", "_clin"), how="inner")

    out = pd.DataFrame({
        "lesion_id": merged["lesion_id"],
        "isic_id_derm": merged["isic_id_derm"],
        "isic_id_clin": merged["isic_id_clin"],
        "y_true": merged["diagnosis_1_derm"],
        "skin_tone_class": merged["skin_tone_class_derm"],
    })
    return out


# =========================
# Evaluation
# =========================
def run_eval(df_lesions, scenario, isic_index, all_files):
    rows = []
    prompt  = build_prompt()
    prompt2 = build_prompt2()

    for _, r in df_lesions.iterrows():
        derm_id = str(r["isic_id_derm"])
        clin_id = str(r["isic_id_clin"])

        derm_path = find_image_path(derm_id, isic_index, all_files)

        if scenario == "derm_only":
            urls = [encode_image_to_data_url(derm_path)]
            pred = call_gpt_with_images(urls, prompt)
        elif scenario == "derm_plus_clin":
            clin_path = find_image_path(clin_id, isic_index, all_files)
            urls = [encode_image_to_data_url(derm_path), encode_image_to_data_url(clin_path)]
            pred = call_gpt_with_images(urls, prompt2)
        else:
            raise ValueError("scenario must be 'derm_only' or 'derm_plus_clin'")

        y_pred = pred.get("pred")
        conf = float(pred.get("confidence", np.nan))

        rows.append({
            "lesion_id": r["lesion_id"],
            "scenario": scenario,
            "y_true": r["y_true"],
            "y_pred": y_pred,
            "confidence": conf,
            "skin_tone_class": r["skin_tone_class"],
            "isic_id_derm": derm_id,
            "isic_id_clin": clin_id,
            "derm_path": str(derm_path),
        })

        time.sleep(SLEEP_S)

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame):
    label_map = {"Malignant": 1, "Benign": 0}
    y_true = results["y_true"].map(label_map).values
    y_pred = results["y_pred"].map(label_map).values

    # Probability malignant using confidence
    prob_mal = np.where(results["y_pred"] == "Malignant",
                        results["confidence"],
                        1 - results["confidence"])

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # row true [Malignant,Benign], col pred [Malignant,Benign]
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    auc = roc_auc_score(y_true, prob_mal) if len(np.unique(y_true)) == 2 else np.nan

    print("Accuracy:", acc)
    print("Sensitivity (Malignant TPR):", sens)
    print("Specificity (Benign TNR):", spec)
    print("AUROC:", auc)
    print("Confusion matrix rows=true [Malignant,Benign], cols=pred [Malignant,Benign]:\n", cm)

    print("\nBy skin_tone_class:")
    for tone, sub in results.groupby("skin_tone_class"):
        yt = sub["y_true"].map(label_map).values
        yp = sub["y_pred"].map(label_map).values
        acc_t = accuracy_score(yt, yp)
        if len(np.unique(yt)) < 2:
            auc_t = np.nan
        else:
            pm = np.where(sub["y_pred"] == "Malignant", sub["confidence"], 1 - sub["confidence"])
            auc_t = roc_auc_score(yt, pm)
        print(f"  tone={tone}  n={len(sub)}  acc={acc_t:.3f}  auc={auc_t if not np.isnan(auc_t) else 'NA'}")


if __name__ == "__main__":
    df_pairs = pd.read_csv(CSV_PATH)
    df_lesions = lesion_level_table(df_pairs)
    print("Lesions to evaluate:", df_lesions.shape[0])

    # Build image index once (fast lookups during eval)
    isic_index, all_files = build_isic_index(IMAGES_ROOT)
    print(f"Indexed {len(all_files)} image files under: {IMAGES_ROOT}")

    # Scenario 1
    res1 = run_eval(df_lesions, "derm_only", isic_index, all_files)
    res1.to_csv("gpt52_milk10k_derm_only_predictions.csv", index=False)
    print("\n=== Dermoscopy only ===")
    summarize(res1)

    # Scenario 2
    res2 = run_eval(df_lesions, "derm_plus_clin", isic_index, all_files)
    res2.to_csv("gpt52_milk10k_derm_plus_clin_predictions.csv", index=False)
    print("\n=== Dermoscopy + Clinical close-up ===")
    summarize(res2)
