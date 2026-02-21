#!/usr/bin/env python3

"""
Evaluating GPT-5.2 for top-3 differential diagnosis on MILK10k

How to run:
  python milk10k_top3_eval.py
  python milk10k_top3_eval.py --images "/path/to/images" --csv "milk10k_500.csv" --limit 50
"""

from __future__ import annotations

import os
import re
import json
import time
import base64
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from openai import OpenAI


# =========================
# DEFAULT CONFIGURATION
# =========================
DEFAULT_CONFIG = {
    "images_root": "../milk10k/images",
    "csv_path": "../script-preprocess/milk10k_500.csv",
    "output_dir": ".",
    "model": "gpt-5.2",
    "truth_col": "diagnosis_2",
    "limit": 0,                  # 0 = all
    "sleep_s": 0.15,
    "max_retries": 3,
    "backoff_base": 1.6,
    "seed": 42,
    "resume": True,
}


# =========================
# JSON parsing
# =========================
def parse_json_safely(text: str) -> Dict[str, Any]:
    """
    Extract and parse a JSON object from possibly messy model output.
    - strips ```json fences
    - extracts first {...} block if extra text exists
    """
    if text is None:
        raise ValueError("Empty response text")
    s = text.strip()

    # Strip markdown fences
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # Try direct JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # Extract first JSON object
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))

    raise ValueError(f"Could not parse JSON. Raw output (first 1200 chars):\n{s[:1200]}")


# =========================
# Image indexing & encoding
# =========================
def build_image_index(images_root: Path) -> Dict[str, Path]:
    """
    Map filename stem -> file path for jpg/jpeg/png recursively.
    Adds convenience keys for ISIC_ prefix variants.
    """
    exts = {".jpg", ".jpeg", ".png"}
    idx: Dict[str, Path] = {}

    for p in images_root.rglob("*"):
        if p.suffix.lower() not in exts:
            continue
        stem = p.stem.strip()
        idx[stem] = p
        idx[stem.upper()] = p
        idx[stem.lower()] = p

        if stem.upper().startswith("ISIC_"):
            idx[stem.upper().replace("ISIC_", "")] = p
            idx[stem.lower().replace("isic_", "")] = p

    return idx


def find_image_path(isic_id: str, idx: Dict[str, Path], images_root: Path) -> Path:
    k = str(isic_id).strip()
    if k in idx:
        return idx[k]
    if k.upper() in idx:
        return idx[k.upper()]
    if k.lower() in idx:
        return idx[k.lower()]

    k2u = f"ISIC_{k}".upper()
    if k2u in idx:
        return idx[k2u]

    # Fallback substring search (slower but robust)
    exts = {".jpg", ".jpeg", ".png"}
    hits = []
    for p in images_root.rglob("*"):
        if p.suffix.lower() in exts and k in p.stem:
            hits.append(p)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        hits.sort(key=lambda x: len(x.stem))
        return hits[0]

    raise FileNotFoundError(f"Could not find image for isic_id={isic_id} under {images_root}")


def to_data_url(img_path: Path) -> str:
    b = img_path.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    mime = "image/jpeg" if img_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64}"


# =========================
# MILK10k lesion pairing
# =========================
def build_lesion_pairs(df: pd.DataFrame, truth_col: str) -> pd.DataFrame:
    """
    Expect 2 rows per lesion_id (derm + clinical). Returns 1 row per lesion:
      lesion_id, isic_id_derm, isic_id_clin, truth, skin_tone_class
    """
    tmp = df.copy()
    tmp["image_type_norm"] = tmp["image_type"].astype(str).str.lower()

    derm = tmp[tmp["image_type_norm"].str.contains("derm")].copy()
    clin = tmp[tmp["image_type_norm"].str.contains("clinic")].copy()

    merged = derm.merge(clin, on="lesion_id", suffixes=("_derm", "_clin"), how="inner")

    # truth_col should exist on both; use derm side
    truth_key = f"{truth_col}_derm" if f"{truth_col}_derm" in merged.columns else truth_col

    out = pd.DataFrame({
        "lesion_id": merged["lesion_id"],
        "isic_id_derm": merged["isic_id_derm"],
        "isic_id_clin": merged["isic_id_clin"],
        "truth": merged[truth_key],
        "skin_tone_class": merged["skin_tone_class_derm"] if "skin_tone_class_derm" in merged.columns else merged.get("skin_tone_class", ""),
    })

    return out


# =========================
# Prompt
# =========================
def build_prompt_open_vocab() -> str:
    return (
        "You are evaluating a skin lesion based on a dermoscopic image (along with clinical close-up if provided).\n"
        "Task: Provide an ordered Top-3 differential diagnosis list (most to least likely)\n"
        "for the lesion shown.\n\n"
        "Return ONLY valid JSON with exactly this key:\n"
        "  differential: [\n"
        "    {\"diagnosis\": \"...\", \"confidence\": 0.0},\n"
        "    {\"diagnosis\": \"...\", \"confidence\": 0.0},\n"
        "    {\"diagnosis\": \"...\", \"confidence\": 0.0}\n"
        "  ]\n"
        "Rules:\n"
        "- Provide exactly 3 items.\n"
        "- 'confidence' must be a number in [0,1] and non-increasing.\n"
        "- Strict JSON only (double quotes). No extra keys. No prose. No code fences."
    )


# =========================
# dx2 mapping (free text -> diagnosis_2)
# =========================
MALIGNANT_DX2 = {
    "malignant melanocytic proliferations melanoma",
    "malignant epidermal proliferations",
    "malignant adnexal epithelial proliferations follicular",
    "collision at least one malignant proliferation",
}

def is_malignant_dx2(norm_dx2_label: str) -> bool:
    return norm_dx2_label in MALIGNANT_DX2
    
def norm_dx2(s: Any) -> str:
    """Normalize MILK10k diagnosis_2 group label (truth) for matching."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    x = str(s).strip().lower()
    #x = re.sub(r"[$begin:math:text$$end:math:text$$begin:math:display$$end:math:display$\{\}]", " ", x)
    #x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def pred_to_dx2_group(pred: Any) -> Optional[str]:
    """
    Map model-predicted free-text diagnosis -> normalized dx2 group string.
    Returns None if unknown/unmapped.
    """
    if pred is None or (isinstance(pred, float) and pd.isna(pred)):
        return None
    p = str(pred).strip().lower()
    p = re.sub(r"[^a-z0-9]+", " ", p)
    p = re.sub(r"\s+", " ", p).strip()

    # malignant melanocytic (melanoma)
    if re.search(r"\b(melanoma|melanoma in situ|lentigo maligna)\b", p):
        return "malignant melanocytic proliferations melanoma"

    # benign melanocytic (nevus)
    if re.search(r"\b(nevus|naevus|mole|melanocytic nevus|intradermal nevus|junctional nevus|compound nevus)\b", p):
        return "benign melanocytic proliferations"
    if re.search(r"\b(dysplastic nevus|atypical nevus)\b", p):
        return "benign melanocytic proliferations"

    # flat pigmentations (lentigo etc.) – separate from nevus
    if re.search(r"\b(lentigo|solar lentigo|lentigines|ephelis|freckle)\b", p):
        return "flat melanotic pigmentations not melanocytic nevus"

    # malignant epidermal (BCC / SCC / AK / Bowen / KA)
    if re.search(r"\b(basal cell carcinoma|bcc)\b", p):
        return "malignant epidermal proliferations"
    if re.search(r"\b(squamous cell carcinoma|scc|bowen)\b", p):
        return "malignant epidermal proliferations"
    if re.search(r"\b(actinic keratosis|ak)\b", p):
        return "malignant epidermal proliferations"
    if re.search(r"\b(keratoacanthoma)\b", p):
        return "malignant epidermal proliferations"

    # benign epidermal (SK, wart, etc.)
    if re.search(r"\b(seborrheic keratosis|seborrheic keratoses|lichenoid keratosis)\b", p):
        return "benign epidermal proliferations"
    if re.search(r"\b(wart|verruca)\b", p):
        return "benign epidermal proliferations"

    # benign soft tissue fibro-histiocytic
    if re.search(r"\b(dermatofibroma|fibrous histiocytoma)\b", p):
        return "benign soft tissue proliferations fibro histiocytic"

    # benign soft tissue vascular
    if re.search(r"\b(hemangioma|angioma|venous lake|pyogenic granuloma|angiokeratoma|vascular)\b", p):
        return "benign soft tissue proliferations vascular"

    # mast cell
    if re.search(r"\b(mastocytoma|mast cell)\b", p):
        return "mast cell proliferations"

    # inflammatory / infectious
    if re.search(r"\b(dermatitis|eczema|psoriasis|tinea|fungal|infection|impetigo|folliculitis)\b", p):
        return "inflammatory or infectious diseases"

    # exogenous
    if re.search(r"\b(tattoo|foreign body|exogenous)\b", p):
        return "exogenous"

    # benign adnexal (rough)
    if re.search(r"\b(poroma|hidradenoma|spiradenoma|syringoma)\b", p):
        return "benign adnexal epithelial proliferations apocrine or eccrine"

    # collisions / other: not reliably inferable from free text without more rules
    return None


def hit_atk_dx2(truth_dx2: Any, pred_texts: List[Any], k: int) -> bool:
    """True if truth dx2 group is among mapped top-k predicted dx2 groups."""
    t = norm_dx2(truth_dx2)
    mapped: List[str] = []
    for p in pred_texts[:k]:
        g = pred_to_dx2_group(p)
        if g is not None:
            mapped.append(norm_dx2(g))
    return bool(t) and (t in mapped)


# =========================
# GPT call
# =========================
def call_gpt_top3_open(
    client: OpenAI,
    model: str,
    prompt: str,
    data_urls: List[str],
    max_retries: int,
    backoff_base: float,
) -> List[Dict[str, Any]]:
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            content = [{"type": "input_text", "text": prompt}]
            for u in data_urls:
                content.append({"type": "input_image", "image_url": u})

            resp = client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
            )

            obj = parse_json_safely(resp.output_text)
            diff = obj.get("differential", [])

            if not isinstance(diff, list) or len(diff) != 3:
                raise ValueError(f"Expected differential list of length 3. Got: {diff}")

            out = []
            for it in diff[:3]:
                out.append({
                    "diagnosis": str(it.get("diagnosis", "")).strip(),
                    "confidence": float(it.get("confidence", 0.0)),
                })
            return out

        except Exception as e:
            last = e
            time.sleep(backoff_base ** attempt)

    raise RuntimeError(f"Failed after {max_retries} retries. Last error: {last}")


# =========================
# Resume helpers
# =========================
def load_done_lesions(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        if "lesion_id" in df.columns:
            return set(df["lesion_id"].astype(str).tolist())
    except Exception:
        return set()
    return set()

def binary_metrics_from_df(df):
    """
    df must contain:
      truth_bin (bool)
      pred1_bin (bool)
    """
    TP = int(((df.truth_bin == True) & (df.pred1_bin == True)).sum())
    TN = int(((df.truth_bin == False) & (df.pred1_bin == False)).sum())
    FP = int(((df.truth_bin == False) & (df.pred1_bin == True)).sum())
    FN = int(((df.truth_bin == True) & (df.pred1_bin == False)).sum())

    eps = 1e-12
    accuracy = (TP + TN) / max(TP + TN + FP + FN, 1)
    sensitivity = TP / max(TP + FN, 1)      # recall for malignant
    specificity = TN / max(TN + FP, 1)
    precision = TP / max(TP + FP, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, eps)

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "confusion_matrix": [[TN, FP],
                             [FN, TP]]
    }
    
# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="MILK10k GPT-5.2 Top-3 evaluation.")
    ap.add_argument("--images", type=Path, default=Path(DEFAULT_CONFIG["images_root"]))
    ap.add_argument("--csv", type=Path, default=Path(DEFAULT_CONFIG["csv_path"]))
    ap.add_argument("--out", type=Path, default=Path(DEFAULT_CONFIG["output_dir"]))
    ap.add_argument("--model", type=str, default=DEFAULT_CONFIG["model"])
    ap.add_argument("--truth-col", type=str, default=DEFAULT_CONFIG["truth_col"])
    ap.add_argument("--limit", type=int, default=DEFAULT_CONFIG["limit"])
    ap.add_argument("--sleep", type=float, default=DEFAULT_CONFIG["sleep_s"])
    ap.add_argument("--max-retries", type=int, default=DEFAULT_CONFIG["max_retries"])
    ap.add_argument("--backoff", type=float, default=DEFAULT_CONFIG["backoff_base"])
    ap.add_argument("--resume", action="store_true", default=DEFAULT_CONFIG["resume"])
    args = ap.parse_args()

    if args.truth_col != "diagnosis_2":
        raise SystemExit("This script maps predictions to MILK10k diagnosis_2 groups; please use --truth-col diagnosis_2.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY env var not set.")
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(args.csv)
    pairs = build_lesion_pairs(df, truth_col=args.truth_col)

    if args.limit and args.limit > 0:
        pairs = pairs.head(args.limit).copy()

    print(f"[info] lesions to evaluate: {len(pairs)}")
    print(f"[info] truth column       : {args.truth_col}")
    print(f"[info] images root        : {args.images.resolve()}")
    print(f"[info] model              : {args.model}")

    # Print unique truth groups (normalized) so you can sanity check label space
    uniq_truth = sorted(set(pairs["truth"].dropna().map(norm_dx2).tolist()))
    print("[check] unique diagnosis_2 (normalized) in this run:")
    for x in uniq_truth[:50]:
        print("  -", x)
    if len(uniq_truth) > 50:
        print(f"  ... ({len(uniq_truth)-50} more)")

    idx = build_image_index(args.images)
    prompt = build_prompt_open_vocab()

    args.out.mkdir(parents=True, exist_ok=True)

    out_derm = args.out / "milk10k_top3_open_dx2_derm_only.csv"
    out_pair = args.out / "milk10k_top3_open_dx2_derm_plus_clin.csv"
    out_metrics = args.out / "milk10k_top3_open_dx2_metrics.json"

    done_derm = load_done_lesions(out_derm) if args.resume else set()
    done_pair = load_done_lesions(out_pair) if args.resume else set()

    def evaluate_scenario(scenario: str, out_csv: Path, done_set: set) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        it = pairs.iterrows()
        if tqdm is not None:
            it = tqdm(it, total=len(pairs), desc=f"Predicting ({scenario})", unit="lesion")

        for _, r in it:
            lesion_id = str(r["lesion_id"])
            if lesion_id in done_set:
                continue

            truth = r["truth"]
            skin_tone = r.get("skin_tone_class", "")

            try:
                derm_path = find_image_path(r["isic_id_derm"], idx, args.images)
                derm_url = to_data_url(derm_path)

                if scenario == "derm_only":
                    urls = [derm_url]
                else:
                    clin_path = find_image_path(r["isic_id_clin"], idx, args.images)
                    clin_url = to_data_url(clin_path)
                    urls = [derm_url, clin_url]

                diff = call_gpt_top3_open(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    data_urls=urls,
                    max_retries=args.max_retries,
                    backoff_base=args.backoff,
                )

                preds = [it.get("diagnosis", "") for it in diff[:3]]
                confs = [float(it.get("confidence", 0.0)) for it in diff[:3]]

                hit1 = hit_atk_dx2(truth, preds, k=1)
                hit3 = hit_atk_dx2(truth, preds, k=3)

                # mapped groups for debugging
                pred_g1 = pred_to_dx2_group(preds[0]) if len(preds) > 0 else None
                pred_g2 = pred_to_dx2_group(preds[1]) if len(preds) > 1 else None
                pred_g3 = pred_to_dx2_group(preds[2]) if len(preds) > 2 else None

                rows.append({
                    "lesion_id": lesion_id,
                    "scenario": scenario,
                    "truth": truth,
                    "truth_dx2_norm": norm_dx2(truth),
                    "top1": preds[0] if len(preds) > 0 else "",
                    "top1_conf": confs[0] if len(confs) > 0 else "",
                    "top2": preds[1] if len(preds) > 1 else "",
                    "top2_conf": confs[1] if len(confs) > 1 else "",
                    "top3": preds[2] if len(preds) > 2 else "",
                    "top3_conf": confs[2] if len(confs) > 2 else "",
                    "pred_dx2_g1": pred_g1,
                    "pred_dx2_g2": pred_g2,
                    "pred_dx2_g3": pred_g3,
                    "top1_correct_dx2": bool(hit1),
                    "hit_at3_dx2": bool(hit3),
                    "skin_tone_class": skin_tone,
                    "isic_id_derm": str(r["isic_id_derm"]),
                    "isic_id_clin": str(r["isic_id_clin"]),
                    "_error": "",
                })

            except Exception as e:
                rows.append({
                    "lesion_id": lesion_id,
                    "scenario": scenario,
                    "truth": truth,
                    "truth_dx2_norm": norm_dx2(truth),
                    "top1": "", "top1_conf": "",
                    "top2": "", "top2_conf": "",
                    "top3": "", "top3_conf": "",
                    "pred_dx2_g1": None,
                    "pred_dx2_g2": None,
                    "pred_dx2_g3": None,
                    "top1_correct_dx2": "",
                    "hit_at3_dx2": "",
                    "skin_tone_class": skin_tone,
                    "isic_id_derm": str(r["isic_id_derm"]),
                    "isic_id_clin": str(r["isic_id_clin"]),
                    "_error": str(e),
                })

            time.sleep(args.sleep)

            # Append incrementally for resume safety (every row)
            if rows:
                df_part = pd.DataFrame(rows)
                if out_csv.exists():
                    df_part.to_csv(out_csv, mode="a", header=False, index=False)
                else:
                    df_part.to_csv(out_csv, index=False)
                rows = []

        return pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()

    # Evaluate scenarios
    df_derm = evaluate_scenario("derm_only", out_derm, done_derm)
    df_pair = evaluate_scenario("derm_plus_clin", out_pair, done_pair)

    def add_binary_labels(df):
        df = df.copy()
        df["truth_bin"] = df["truth_dx2_norm"].apply(is_malignant_dx2)
        df["pred1_bin"] = df["pred_dx2_g1"].fillna("").map(norm_dx2).apply(is_malignant_dx2)
        return df

    # Summaries
    def summarize(dfres: pd.DataFrame, name: str) -> Dict[str, Any]:
        # eval rows: no error + has truth + has top1
        err = dfres["_error"]
        ok = dfres[ (err.isna() | (err.astype(str).str.strip() == "")) &
                    (dfres["truth"].astype(str).str.strip() != "") &
                    (dfres["top1"].astype(str).str.strip() != "")]
        out = {
            "scenario": name,
            "n_total": int(len(dfres)),
            "n_eval": int(len(ok)),
            "top1_accuracy_dx2": None,
            "hit_at3_dx2": None,
            "by_skin_tone_class": {},
            "binary_metrics": None,
            "mapping_coverage_top1": None,
        }
        if len(ok) == 0:
            return out

        out["top1_accuracy_dx2"] = float((ok["top1_correct_dx2"] == True).mean())
        out["hit_at3_dx2"] = float((ok["hit_at3_dx2"] == True).mean())
        out["mapping_coverage_top1"] = float(ok["pred_dx2_g1"].notna().mean())

        # ---- Binary metrics (overall) ----
        ok2 = add_binary_labels(ok)
        out["binary_metrics"] = binary_metrics_from_df(ok2)
    
        # by skin tone
        for tone, sub in ok.groupby("skin_tone_class"):
            sub2 = add_binary_labels(sub)
            out["by_skin_tone_class"][str(tone)] = {
                "n": int(len(sub)),
                "top1_accuracy_dx2": float((sub["top1_correct_dx2"] == True).mean()),
                "hit_at3_dx2": float((sub["hit_at3_dx2"] == True).mean()),
                "mapping_coverage_top1": float(sub["pred_dx2_g1"].notna().mean()),
                "binary_metrics": binary_metrics_from_df(sub2),
            }
        return out

    m1 = summarize(df_derm, "derm_only")
    m2 = summarize(df_pair, "derm_plus_clin")

    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump({"derm_only": m1, "derm_plus_clin": m2}, f, indent=2)

    #print("\n=== Dermoscopy only ===")
    #print(json.dumps(m1, indent=2))
    #print("\n=== Dermoscopy + Clinical closeup ===")
    #print(json.dumps(m2, indent=2))
    def print_compact_metrics(name, metrics):
        print(f"\n=== {name} ===")
        if metrics.get("binary_metrics") is None:
            print("No valid evaluations.")
            return

        out = {
            "binary_metrics": metrics["binary_metrics"],
            "mapping_coverage_top1": metrics.get("mapping_coverage_top1"),
        }
        print(json.dumps(out, indent=2))


    print_compact_metrics("Dermoscopy only", m1)
    print_compact_metrics("Dermoscopy + Clinical", m2)

    print("\nSaved:")
    print(f"  {out_derm}")
    print(f"  {out_pair}")
    print(f"  {out_metrics}")


if __name__ == "__main__":
    main()
