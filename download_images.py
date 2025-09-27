#!/usr/bin/env python3
"""
Fetch ISIC metadata (single summary table) + images for a list of IDs.

Examples:
  python download_isic.py isic-100-image-ids.txt --out isic_data --xlsx
  python download_isic.py ham10k-500-image-ids.txt --out ham10k_images --xlsx
  # python download_isic.py isic-100-image-ids.txt --id-col isic_id --out isic_out --workers 8 --skip-existing
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

API_BASE = "https://api.isic-archive.com/api/v2/images"
UA = {"User-Agent": "ISIC-batch-fetch/1.0 (+research use)"}

SUGGESTED_FIRST_COLUMNS = [
    "isic_id",
    "name",
    "created",
    "updated",
    "image.width",
    "image.height",
    "metadata.diagnosis",
    "metadata.benign_malignant",
    "metadata.anatom_site_general",
    "metadata.patient_id",
    "_status",
    "_image_path",
]

def s3_url_for(isic_id: str, ext: str) -> str:
    return f"https://isic-archive.s3.amazonaws.com/images/{isic_id}.{ext}"

def fetch_image_bytes(isic_id: str, session: requests.Session) -> tuple[bytes, str]:
    """
    Try S3 JPG, then S3 PNG, then API /download. Return (bytes, 's3_jpg'|'s3_png'|'api_download').
    Raise on failure.
    """
    # 1) S3 JPG
    url = s3_url_for(isic_id, "jpg")
    r = session.get(url, headers=UA, timeout=60)
    if r.ok and r.content:
        return r.content, "s3_jpg"

    # 2) S3 PNG (some items are PNG)
    url = s3_url_for(isic_id, "png")
    r = session.get(url, headers=UA, timeout=60)
    if r.ok and r.content:
        return r.content, "s3_png"

    # 3) API download fallback
    url = f"{API_BASE}/{isic_id}/download"
    r = session.get(url, headers=UA, timeout=60, allow_redirects=True)
    r.raise_for_status()
    if not r.content:
        raise RuntimeError("Empty content from API download")
    return r.content, "api_download"

def read_ids(path: Path, id_col: str | None) -> List[str]:
    if path.suffix.lower() == ".csv":
        if not id_col:
            raise ValueError("--id-col is required for CSV input")
        ids: List[str] = []
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                v = (row.get(id_col) or "").strip()
                if v:
                    ids.append(v)
    else:
        ids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # de-dupe, preserve order
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def flatten(d: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flatten dicts/lists: {"a":{"b":1},"c":[10]} -> {"a.b":1,"c[0]":10}"""
    items: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten(v, new_key, sep=sep))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{parent_key}[{i}]"
            items.update(flatten(v, new_key, sep=sep))
    else:
        items[parent_key] = d
    return items

def get_json(url: str, session: requests.Session, retries=3, backoff=1.6):
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, headers=UA, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 2))
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(backoff ** attempt)
    raise last

def get_binary(url: str, session: requests.Session, retries=3, backoff=1.6):
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, headers=UA, timeout=60, stream=True)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 2))
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
            time.sleep(backoff ** attempt)
    raise last

def fetch_one(isic_id: str, out_dir: Path, session: requests.Session, image_ext: str,
              polite_delay: float, skip_existing: bool) -> Tuple[Dict[str, Any], str | None]:
    """
    Fetch metadata and image for one ID.
    Returns: (row_dict, failure_message_or_None)
    """
    time.sleep(polite_delay)  # be nice
    meta_url = f"{API_BASE}/{isic_id}"
    img_url = f"{API_BASE}/{isic_id}/download"

    row: Dict[str, Any] = {"isic_id": isic_id, "_status": "", "_image_path": ""}

    # 1) Metadata
    try:
        meta = get_json(meta_url, session=session)
        flat = flatten(meta)
        row.update(flat)
        row["_status"] = "OK"
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", "NA")
        row["_status"] = f"HTTPError {code}"
        # For 404 we still attempt image (sometimes meta or image may be independently unavailable),
        # but typically image will fail too. We'll continue and record results.
    except Exception as e:
        row["_status"] = f"Error: {e}"

    # 2) Image
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Default to JPG filename; if we end up with PNG from S3, we’ll rename.
    img_path = img_dir / f"{isic_id}.jpg"

    if skip_existing and (img_dir / f"{isic_id}.jpg").exists():
        row["_image_path"] = str(img_dir / f"{isic_id}.jpg")
        row["_image_src"] = "existing"
    elif skip_existing and (img_dir / f"{isic_id}.png").exists():
        row["_image_path"] = str(img_dir / f"{isic_id}.png")
        row["_image_src"] = "existing"
    else:
        try:
            img_bytes, src = fetch_image_bytes(isic_id, session)
            # If S3 said PNG, switch extension
            if src == "s3_png":
                img_path = img_dir / f"{isic_id}.png"
            img_path.write_bytes(img_bytes)
            row["_image_path"] = str(img_path)
            row["_image_src"] = src
        except Exception as e:
            if row["_status"] == "OK":
                row["_status"] = f"OK (image error: {e})"
            else:
                row["_status"] = f"{row['_status']} (image error: {e})"
            row["_image_src"] = ""

    return row, (None if row["_status"].startswith("OK") else row["_status"])

def main():
    ap = argparse.ArgumentParser(description="Download ISIC images + single metadata table.")
    ap.add_argument("ids_file", type=Path, help="TXT (one per line) or CSV of IDs.")
    ap.add_argument("--id-col", type=str, default=None, help="Column name if input is CSV.")
    ap.add_argument("--out", type=Path, default=Path("isic_out"), help="Output directory.")
    ap.add_argument("--workers", type=int, default=6, help="Parallel workers.")
    ap.add_argument("--image-ext", choices=["jpg", "jpeg", "png"], default="jpg", help="Image file extension.")
    ap.add_argument("--xlsx", action="store_true", help="Also save an Excel copy of the table.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip downloading images that already exist.")
    ap.add_argument("--polite-delay", type=float, default=0.1, help="Per-request delay (seconds). Increase if rate-limited.")
    args = ap.parse_args()

    try:
        ids = read_ids(args.ids_file, args.id_col)
    except Exception as e:
        print(f"Failed to read IDs: {e}", file=sys.stderr)
        sys.exit(1)

    if not ids:
        print("No IDs found.", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(UA)

    rows: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    print(f"Processing {len(ids)} IDs with {args.workers} workers…")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                fetch_one,
                isic_id,
                args.out,
                session,
                args.image_ext,
                args.polite_delay,
                args.skip_existing,
            ): isic_id
            for isic_id in ids
        }
        for fut in as_completed(futures):
            isic_id = futures[fut]
            try:
                row, fail = fut.result()
                rows.append(row)
                if fail:
                    failures.append((isic_id, fail))
                    print(f"[WARN] {isic_id} -> {fail}", file=sys.stderr)
                else:
                    print(f"[OK]   {isic_id}")
            except Exception as e:
                failures.append((isic_id, str(e)))
                rows.append({"isic_id": isic_id, "_status": f"Fatal: {e}", "_image_path": ""})
                print(f"[FAIL] {isic_id} -> {e}", file=sys.stderr)

    # Build table
    df = pd.DataFrame(rows).fillna("")
    # Put suggested columns first
    ordered_cols = []
    for c in SUGGESTED_FIRST_COLUMNS:
        if c in df.columns and c not in ordered_cols:
            ordered_cols.append(c)
    for c in df.columns:
        if c not in ordered_cols:
            ordered_cols.append(c)
    df = df[ordered_cols]

    # Save outputs
    meta_csv = args.out / "isic_metadata.csv"
    df.to_csv(meta_csv, index=False, encoding="utf-8")
    print(f"\nSaved CSV: {meta_csv}")
    if args.xlsx:
        meta_xlsx = args.out / "isic_metadata.xlsx"
        df.to_excel(meta_xlsx, index=False)
        print(f"Saved Excel: {meta_xlsx}")

    if failures:
        fail_path = args.out / "failures.log"
        with fail_path.open("w", encoding="utf-8") as f:
            for i, msg in failures:
                f.write(f"{i}\t{msg}\n")
        print(f"Some IDs had issues; see: {fail_path}")

if __name__ == "__main__":
    main()
