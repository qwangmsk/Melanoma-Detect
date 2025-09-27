#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# My command: python isic_top3_eval.py --images isic_images --meta isic_metadata.xlsx --sheet "Sheet1" --out isic_out/preds-t3 --model gpt-5 --truth-col "metadata.clinical.diagnosis_1"
# My command: python isic_top3_eval.py --images ham10k_images --meta ham10k_images/isic_metadata.xlsx --sheet "Sheet1" --out ham10k_out/preds-t3 --model gpt-5 --truth-col "metadata.clinical.diagnosis_1"

from __future__ import annotations
import os, sys, re, json, time, base64, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from sklearn.metrics import accuracy_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False

from openai import OpenAI

CLASSES = ["melanoma", "melanocytic nevus"]
INDETERMINATE_TOKENS = {"indeterminate"}
ISIC_ID_RE = re.compile(r"ISIC_\d+", re.IGNORECASE)

# -------- I/O helpers --------
def load_metadata(path: Path, sheet: str | None) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        sheet_arg = 0 if (sheet is None or str(sheet).strip() == "") else sheet
        tmp = pd.read_excel(path, sheet_name=sheet_arg, dtype=str)
        if isinstance(tmp, dict):
            first_key = next(iter(tmp.keys()))
            df = tmp[first_key]
        else:
            df = tmp
        return df.fillna("")
    elif ext == ".csv":
        return pd.read_csv(path, dtype=str).fillna("")
    else:
        raise SystemExit("ERROR: Unsupported metadata format. Use .xlsx, .xls, or .csv")

def discover_images(images_dir: Path, limit: int = 0) -> List[Path]:
    if not images_dir.exists():
        raise SystemExit(f"ERROR: Images folder not found: {images_dir}")
    img_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    files = [p for p in images_dir.rglob("*") if p.suffix in img_exts]
    files = [p for p in files if ISIC_ID_RE.search(p.name)]
    files.sort()
    if limit and limit > 0:
        files = files[:limit]
    return files

def extract_isic_id(name: str) -> str | None:
    m = ISIC_ID_RE.search(name)
    return m.group(0).upper() if m else None

# -------- Truth normalization --------
def canon_truth(raw: str) -> str:
    t = (raw or "").strip().lower()
    if not t:
        return ""
    if any(tok in t for tok in INDETERMINATE_TOKENS):
        return "melanocytic nevus"
    if "melanoma" in t or "malignant" in t:
        return "melanoma"
    if "nevus" in t or "naevus" in t or "benign" in t:
        return "melanocytic nevus"
    return ""

# -------- OpenAI helpers (Chat Completions, JSON mode) --------
def b64_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"

def extract_json_from_chat(resp) -> Dict[str, Any]:
    # Most SDK versions: the content is a JSON string (thanks to response_format json_object)
    txt = resp.choices[0].message.content
    try:
        return json.loads(txt)
    except Exception:
        # last resort: pull the first {...} blob
        a, b = txt.find("{"), txt.rfind("}")
        if a != -1 and b != -1 and b > a:
            return json.loads(txt[a:b+1])
        raise ValueError("Could not parse JSON from chat completion output")

def call_gpt_top3_binary_chat(client: OpenAI, model: str, image_path: Path,
                              retries: int = 3, backoff: float = 1.7) -> List[Dict[str, Any]]:
    """
    Chat Completions (vision) + JSON mode.
    Returns list of 3 dicts: {diagnosis, confidence, rationale?}
    """
    data_url = b64_data_url(image_path)
    prompt = (
        "Provide a ranked differential diagnosis, listing three potential diagnoses from most to least likely based on this dermoscopic image. "
        "Return a JSON object with key 'differential' = array of exactly 3 items ordered from most likely to least likely; "
        "each item must include: diagnosis (string), confidence (0.0-1.0), and optionally a brief rationale."
    )
    last = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                #temperature=0,
                response_format={"type": "json_object"},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
            )
            #print("API responded with model:", getattr(resp, "model", "<unknown>"))
            obj = extract_json_from_chat(resp)
            return obj["differential"]
        except Exception as e:
            last = e
            time.sleep(backoff ** attempt)
    raise last

# -------- Main --------
def main():
    ap = argparse.ArgumentParser(description="Binary Top-3 (Melanoma vs Melanocytic Nevus) via Chat Completions (JSON mode).")
    ap.add_argument("--images", type=Path, required=True, help="Folder containing ISIC images (searches recursively).")
    ap.add_argument("--meta", type=Path, required=True, help="Metadata file (.xlsx/.xls/.csv). Must include 'isic_id'.")
    ap.add_argument("--sheet", type=str, default=None, help="Excel sheet name or index; default first sheet.")
    ap.add_argument("--out", type=Path, default=Path("isic_out/top3_binary"), help="Output folder.")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="Vision-capable model (e.g., gpt-4o-mini, gpt-4o).")
    ap.add_argument("--truth-col", type=str, default="metadata.clinical.diagnosis_1",
                    help="Metadata column for ground truth (e.g., diagnosis_1 or diagnosis_3).")
    ap.add_argument("--limit", type=int, default=0, help="Optionally cap number of images (0=all).")
    args = ap.parse_args()

    print(f"[info] images dir : {args.images.resolve()}")
    print(f"[info] metadata   : {args.meta.resolve()}")
    print(f"[info] sheet      : {args.sheet if args.sheet is not None else '(first sheet)'}")
    print(f"[info] truth col  : {args.truth_col}")
    print(f"[info] model      : {args.model}")

    meta = load_metadata(args.meta, args.sheet)
    if "isic_id" not in meta.columns:
        raise SystemExit("ERROR: metadata must contain 'isic_id' column.")
    if args.truth_col not in meta.columns:
        raise SystemExit(f"ERROR: column '{args.truth_col}' not found in metadata. "
                         f"Available columns (first 25): {', '.join(list(meta.columns)[:25])}")

    print(f"[info] metadata rows: {len(meta)}")
    truth_map = {str(r["isic_id"]).strip().upper(): str(r[args.truth_col]) for _, r in meta.iterrows()}
    meta_ids = set(truth_map.keys())

    files = discover_images(args.images, args.limit)
    print(f"[info] discovered files (by ext & ID pattern): {len(files)}")

    paired: List[Tuple[Path, str]] = []
    for p in files:
        iid = extract_isic_id(p.name)
        if iid and iid.upper() in meta_ids:
            paired.append((p, iid.upper()))
    print(f"[info] files matched to metadata ids         : {len(paired)}")
    for i, (p, iid) in enumerate(paired[:5]):
        print(f"  - sample[{i}]: {p} -> {iid}")

    if not paired:
        print("\n[help] Zero matched images.\n"
              "  • Ensure filenames contain an ISIC id like 'ISIC_0013494.jpg'\n"
              "  • Confirm the images folder path\n"
              "  • Verify metadata 'isic_id' values match filenames (case-insensitive)\n")
        raise SystemExit("ERROR: No images matched. Aborting before inference.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: Set OPENAI_API_KEY in your environment.")
    client = OpenAI(api_key=api_key)

    rows: List[Dict[str, Any]] = []
    for img_path, isic_id in tqdm(paired, desc="Predicting", unit="img"):
        truth_raw = truth_map.get(isic_id, "")
        truth_bin = canon_truth(truth_raw)  # '' if not evaluable

        try:
            diff = call_gpt_top3_binary_chat(client, args.model, img_path)
            # normalize predictions to the two classes
            preds = []
            for item in diff[:3]:
                d_raw = (item.get("diagnosis") or "").lower()
                c = float(item.get("confidence", 0.0))
                d2 = "melanoma" if "melanoma" in d_raw else "melanocytic nevus"
                preds.append((d2, c))
            while len(preds) < 3:
                preds.append(("melanocytic nevus", 0.0))

            (p1, c1), (p2, c2), (p3, c3) = preds[0], preds[1], preds[2]
            hit1 = truth_bin != "" and p1 == truth_bin
            hit3 = truth_bin != "" and truth_bin in {p1, p2, p3}

            rows.append({
                "isic_id": isic_id,
                "truth": truth_bin,
                "top1": p1, "top1_conf": c1,
                "top2": p2, "top2_conf": c2,
                "top3": p3, "top3_conf": c3,
                "top1_correct": bool(hit1),
                "hit_at3": bool(hit3),
            })

        except Exception as e:
            rows.append({
                "isic_id": isic_id,
                "truth": truth_bin,
                "top1": "", "top1_conf": "",
                "top2": "", "top2_conf": "",
                "top3": "", "top3_conf": "",
                "top1_correct": "", "hit_at3": "",
                "_error": str(e),
            })

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).fillna("")
    df.to_csv(out / "isic_top3_binary_summary.csv", index=False)

    # Evaluate (robust indexing)
    for col in ["truth", "top1", "top1_correct", "hit_at3"]:
        if col not in df.columns:
            df[col] = ""
    eval_df = df[(df["truth"].astype(str) != "") & (df["top1"].astype(str) != "")]
    metrics = {
        "n_total": int(len(df)),
        "n_eval": int(len(eval_df)),
        "top1_accuracy": None,
        "hit_at_3": None,
    }
    if not eval_df.empty:
        metrics["top1_accuracy"] = float((eval_df["top1_correct"] == True).mean())
        metrics["hit_at_3"] = float((eval_df["hit_at3"] == True).mean())
        if HAVE_SK:
            try:
                metrics["top1_accuracy_sklearn"] = float(
                    accuracy_score(eval_df["truth"].tolist(), eval_df["top1"].tolist())
                )
            except Exception:
                pass

    with (out / "isic_top3_binary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(out / "isic_top3_binary_metrics.csv", index=False)

    print(f"\nSaved summary CSV : {out / 'isic_top3_binary_summary.csv'}")
    print(f"Saved metrics JSON: {out / 'isic_top3_binary_metrics.json'}")
    print(f"Saved metrics CSV : {out / 'isic_top3_binary_metrics.csv'}")
    if metrics["n_eval"] == 0:
        print("\n[Hint] No evaluable rows. Check truth column values "
              "(must contain 'malignant/melanoma' or 'benign/nevus'), and filename/ID matching.", file=sys.stderr)

if __name__ == "__main__":
    main()
