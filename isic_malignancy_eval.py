#!/usr/bin/env python3
# My command:
# python isic_malignancy_eval.py --images isic_images   --meta isic_images/isic_metadata.xlsx   --sheet "Sheet1" --out isic_out/preds --model gpt-5 --truth-col "metadata.clinical.diagnosis_1"
# python isic_malignancy_eval.py --images ham10k_images --meta ham10k_images/isic_metadata.xlsx --sheet "Sheet1" --out ham10k_out --model gpt-5 --truth-col "metadata.clinical.diagnosis_1"

from __future__ import annotations
import os, sys, base64, json, time, argparse, re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from tqdm import tqdm

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    HAVE_SK = True
except Exception:
    HAVE_SK = False

from openai import OpenAI

# ---------- OpenAI helpers ----------

def _extract_json_from_responses(resp):
    # Try the most robust fields across SDK versions:
    # 1) structured outputs
    if hasattr(resp, "parsed") and resp.parsed:
        return resp.parsed
    # 2) aggregated text
    if hasattr(resp, "output_text") and resp.output_text:
        return json.loads(resp.output_text)
    # 3) raw "output"
    try:
        content = resp.output[0].content[0].text
        return json.loads(content)
    except Exception:
        pass
    # 4) last resort: full JSON dump -> find first JSON object
    s = str(resp)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start:end+1])
    raise ValueError("Could not parse JSON from Responses API output")

def call_gpt_image_binary(client: OpenAI, model: str, image_path: Path,
                          fallback_model: str = "gpt-5",
                          retries=3, backoff=1.7) -> dict:
    """
    Try Responses API with json_schema; if that fails, fall back to Chat Completions JSON mode.
    Returns dict: {"is_melanoma": bool, "likelihood": float, "rationale": str}
    """
    def b64_data_url(p: Path):
        mime = "image/png" if p.suffix.lower()==".png" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('utf-8')}"

    prompt = (
        "Classify this lesion as melanoma or not. If uncertain, still decide but lower likelihood."
        "Return strict JSON matching {is_melanoma:boolean, likelihood:number[0..1], rationale:string}. "
    )
    data_url = b64_data_url(image_path)

    last = None
    for attempt in range(1, retries+1):
        try:
            # --- Preferred: Responses API with structured outputs ---
            resp = client.responses.create(
                model=model,
                # temperature=0,   
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "MelanomaClassification",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "is_melanoma": {"type": "boolean"},
                                "likelihood": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "rationale": {"type": "string"}
                            },
                            "required": ["is_melanoma","likelihood","rationale"],
                            "additionalProperties": False
                        }
                    }
                },
            )
            return _extract_json_from_responses(resp)
        except Exception as e:
            last = e
            time.sleep(backoff**attempt)

    # --- Fallback: Chat Completions with JSON mode + image ---
    # Requires a vision-capable model and JSON mode support.
    for attempt in range(1, retries+1):
        try:
            resp = client.chat.completions.create(
                model=fallback_model,
                response_format={"type": "json_object"},
                messages=[{
                    "role":"user",
                    "content":[
                        {"type":"text","text":prompt},
                        {"type":"image_url","image_url":{"url": data_url}},
                    ]
                }],
            )
            txt = resp.choices[0].message.content
            return json.loads(txt)
        except Exception as e:
            last = e
            time.sleep(backoff**attempt)

    raise last
    
def b64_data_url(image_path: Path) -> str:
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    data = image_path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"

PRED_SCHEMA: Dict[str, Any] = {
    "name": "MelanomaClassification",
    "description": "Binary melanoma classification of a dermoscopic lesion with likelihood and brief rationale.",
    "schema": {
        "type": "object",
        "properties": {
            "is_melanoma": {"type": "boolean"},
            "likelihood":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale":   {"type": "string"}
        },
        "required": ["is_melanoma", "likelihood", "rationale"],
        "additionalProperties": False
    }
}

def normalize_truth_row(row: pd.Series, force_col: str | None = None) -> str:
    # If user forces a column, use it
    if force_col and force_col in row:
        text = str(row[force_col]).strip()
        tl = text.lower()
        if "indeterminate" in tl:
            return "non-melanoma"  # include in scoring
        if "malignant" in tl or "melanoma" in tl:
            return "melanoma"
        if "benign" in tl:
            return "non-melanoma"
        return ""

    # Otherwise your previous logic (keep your existing candidates)…
    # But add this fast path if the clinical column exists:
    if "metadata.clinical.diagnosis_1" in row and str(row["metadata.clinical.diagnosis_1"]).strip():
        tl = str(row["metadata.clinical.diagnosis_1"]).strip().lower()
        if "indeterminate" in tl:
            return "non-melanoma"
        if "malignant" in tl or "melanoma" in tl:
            return "melanoma"
        if "benign" in tl:
            return "non-melanoma"

    # … fallback to your previous heuristics afterwards
    return ""

# ---------- IO helpers ----------
def load_metadata(path: Path, sheet: str | None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet, dtype=str).fillna("")
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str).fillna("")
    else:
        raise ValueError("Unsupported metadata format. Use .xlsx, .xls, or .csv")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="GPT-5 melanoma vs non-melanoma; compact CSV + metrics (reads Excel or CSV).")
    ap.add_argument("--images", type=Path, required=True, help="Folder with ISIC images.")
    ap.add_argument("--meta", type=Path, required=True, help="Path to isic_metadata.xlsx (or CSV).")
    ap.add_argument("--sheet", type=str, default=None, help="Excel sheet name if applicable.")
    ap.add_argument("--out", type=Path, default=Path("isic_out/preds"), help="Output folder.")
    ap.add_argument("--model", type=str, default="gpt-5", help="Vision-capable model name.")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of images.")
    ap.add_argument("--truth-col", type=str, default=None, help="Force a metadata column as ground truth.")
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY", file=sys.stderr); sys.exit(1)

    client = OpenAI(api_key=api_key)

    meta_df = load_metadata(args.meta, args.sheet)
    if "isic_id" not in meta_df.columns:
        print("ERROR: metadata must contain 'isic_id'", file=sys.stderr); sys.exit(1)

    # Collect images
    exts = (".jpg", ".jpeg", ".png")
    files = [p for p in args.images.glob("*") if p.suffix.lower() in exts]
    files = [p for p in files if re.search(r"ISIC_\d+\.(png|jpe?g)$", p.name, re.I)]
    files.sort()
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print("No images found in", args.images, file=sys.stderr); sys.exit(1)

    # Run predictions
    preds: List[Dict[str, Any]] = []
    for img_path in tqdm(files, desc="Classifying", unit="img"):
        isic_id = img_path.stem
        try:
            r = call_gpt_image_binary(client, args.model, img_path)
            label = "melanoma" if r.get("is_melanoma") else "non-melanoma"
            preds.append({
                "isic_id": isic_id,
                "pred_label": label,
                "pred_is_melanoma": bool(r.get("is_melanoma")),
                "pred_likelihood": float(r.get("likelihood", 0.0)),
                "pred_rationale": r.get("rationale", ""),
            })
        except Exception as e:
            preds.append({
                "isic_id": isic_id,
                "pred_label": "",
                "pred_is_melanoma": "",
                "pred_likelihood": "",
                "pred_rationale": "",
                "_error": str(e),
            })

    pred_df = pd.DataFrame(preds).fillna("")
    merged = meta_df.merge(pred_df, on="isic_id", how="left")

    # Ground truth & correctness
    merged["truth_label"] = merged.apply(lambda r: normalize_truth_row(r, args.truth_col), axis=1)
    merged["correct"] = (merged["pred_label"] != "") & (merged["truth_label"] != "") & (merged["pred_label"] == merged["truth_label"])

    # Compact CSV
    compact_cols = ["isic_id", "pred_label", "pred_is_melanoma", "pred_likelihood", "truth_label", "correct", "pred_rationale"]
    compact = merged[compact_cols]
    args.out.mkdir(parents=True, exist_ok=True)
    compact_csv = args.out / "isic_melanoma_summary.csv"
    compact.to_csv(compact_csv, index=False)

    # Full CSV
    full_csv = args.out / "isic_melanoma_summary_full.csv"
    merged.to_csv(full_csv, index=False)

    # Metrics
    eval_rows = merged[(merged["truth_label"] != "") & (merged["pred_label"] != "")]
    metrics = {
        "n_total": len(merged),
        "n_with_truth": int((merged["truth_label"] != "").sum()),
        "n_with_pred": int((merged["pred_label"] != "").sum()),
        "n_evaluated": len(eval_rows),
    }

    if HAVE_SK and not eval_rows.empty:
        y_true = (eval_rows["truth_label"] == "melanoma").astype(int)
        y_pred = (eval_rows["pred_label"] == "melanoma").astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        metrics.update({
            "accuracy": acc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": {"labels": ["non-melanoma","melanoma"], "matrix": cm}
        })
    else:
        metrics["note"] = "Install scikit-learn OR ensure truth/pred both present."

    with (args.out / "isic_melanoma_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved compact CSV: {compact_csv}")
    print(f"Saved full CSV:    {full_csv}")
    print(f"Saved metrics:     {args.out / 'isic_melanoma_metrics.json'}")
    if metrics["n_evaluated"] == 0:
        print("\n[Hint] Ground truth not found. Try --truth-col metadata.benign_malignant "
              "or open the FULL CSV to locate the correct label column.", file=sys.stderr)

if __name__ == "__main__":
    main()
