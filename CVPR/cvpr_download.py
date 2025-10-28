#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

功能：
  - 从 CSV 中筛选 abstract/title 含“多模态”相关表达的行；
  - 下载其 pdf_url 指向的 PDF；
  - 规范化重命名文件，并加上形如：([C]-2025-CVPR) 的前缀。

用法示例：
  python filter_and_download_multimodal_pdfs.py \
      --csv cvpr2025.csv \
      --outdir ./pdfs \
      --year 2025 \
      --conf CVPR \
      --prefix "([C]-{year}-{conf})" \
      --overwrite

参数说明：
  --csv         输入 CSV 文件路径（需包含列：title、abstract、pdf_url；大小写不敏感）。
  --outdir      输出目录，默认 ./downloads
  --year        前缀中的年份，默认 2025
  --conf        前缀中的会议简称，默认 CVPR（可改为 ICCV / ECCV / 其它）
  --prefix      前缀模板，默认 "([C]-{year}-{conf})"
  --overwrite   已存在同名文件时是否覆盖
  --dry-run     只打印将要下载与重命名的信息，不真正下载
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from urllib.parse import urlparse

import unicodedata
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Multimodal-Downloader/1.0; +https://example.org/academic-use)",
    "Accept": "application/pdf, */*;q=0.1",
}

MULTIMODAL_PATTERN = r"""
(?:
    \b(?:multi|cross)[-\s_]?modal(?:ity|ities)?\b
  | \b(?:multi|cross)[-\s_]?modal\b              # multi-modal（不带后缀）
  | \bcrossmodal\b                                # crossmodal（紧连）
  | \bmultimodal(?:ity|ities)?\b                 # multimodal/multimodality/multimodalities
  | \b(?:modal|modality)[-\s_]*fusion\b
  | \bvision[-\s_]?language\b                    # vision-language / vision_language / vision language
)
"""

SEMSEG_PATTERN = r"""
(?:
    \bsemantic[-\s_]?segmentation\b              # semantic segmentation（连字符/下划线/空格）
  | \bsemantic[-\s_]?seg\b                       # semantic-seg / semantic_seg / semantic seg
  | \bsem[-\s_]?seg(?:mentation)?\b              # semseg / sem-seg / sem_seg / sem segmentation
  | \bsegmentation\b
)
"""

MULTI_SEMSEG_PATTERN = r"""
(?:
    (?=.*\b(?:multi|cross)[-\s_]?modal(?:ity|ities)?\b
        | \b(?:modal|modality)[-\s_]*fusion\b
        | \bvision[-\s_]?language\b) 
    (?=.*\bsemantic[-\s_]?segmentation\b
        | \bsem[-\s_]?seg(?:mentation)?\b) 
)
"""

FLAGS = re.IGNORECASE | re.VERBOSE | re.UNICODE
MULTIMODAL_REGEX_1   = re.compile(MULTIMODAL_PATTERN, FLAGS)
SEMSEG_REGEX_2       = re.compile(SEMSEG_PATTERN, FLAGS)
MULTI_SEMSEG_REGEX_3 = re.compile(MULTI_SEMSEG_PATTERN, FLAGS)

MULTIMODAL_REGEX = MULTIMODAL_REGEX_1


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    return s.strip()


def contains_multimodal(s: str) -> bool:
    s = normalize_text(s)
    if not s:
        return False
    return MULTIMODAL_REGEX.search(s) is not None


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def pick_filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = os.path.basename(path) or "paper.pdf"
    return name


def strip_author_prefix(name_wo_ext: str) -> str:
    if "_" in name_wo_ext:
        first, rest = name_wo_ext.split("_", 1)
        if len(rest) >= 15 and (first[:1].isupper() or len(first) <= 25):
            return rest
    return name_wo_ext


def sanitize_title_for_filename(title: str) -> str:
    if title is None:
        title = ""
    title = str(title)
    title = re.sub(r'[\\/:*?"<>|]+', " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _download_once(url: str, outpath: str, overwrite: bool = False, timeout: int = 30) -> Tuple[bool, Optional[str]]:
    """
    单次下载尝试。返回 (是否成功, 失败原因或 None)。
    """
    if not url or str(url).lower() == "nan":
        return False, "empty_url"

    if os.path.exists(outpath) and not overwrite:
        return True, None

    try:
        with requests.get(url, stream=True, headers=HEADERS, timeout=timeout) as r:
            if r.status_code != 200:
                return False, f"http_{r.status_code}"
            tmp = outpath + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, outpath)
            return True, None
    except Exception as e:
        return False, str(e)


def download_pdf(
    url: str,
    outpath: str,
    overwrite: bool = False,
    timeout: int = 60,
    retries: int = 3,
    backoff_sec: float = 1.0,
) -> Tuple[bool, Optional[str], int]:
    last_reason = None
    attempts = 0
    for attempt in range(1, retries + 1):
        attempts = attempt
        ok, reason = _download_once(url, outpath, overwrite=overwrite, timeout=timeout)
        if ok:
            return True, None, attempts
        last_reason = reason
        time.sleep(backoff_sec * (2 ** (attempt - 1)))
    return False, last_reason, attempts


def main():
    ap = argparse.ArgumentParser(description="Filter by multimodal keywords and download PDFs with normalized names.")
    ap.add_argument("--csv", default="./files/cvpr2025_full.csv", help="Input CSV path (must contain columns: title, abstract, pdf_url).")
    ap.add_argument("--outdir", default="./downloads", help="Directory to save PDFs.")
    ap.add_argument("--year", type=int, default=2025, help="Year used in prefix (default: 2025).")
    ap.add_argument("--conf", type=str, default="CVPR", help="Conference token used in prefix (default: CVPR).")
    ap.add_argument(
        "--prefix",
        type=str,
        default="([C]-{year}-{conf})",
        help="Prefix template, supports {year} and {conf}. Default: '([C]-{year}-{conf})'",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without downloading.")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    df = pd.read_csv(args.csv)

    colmap: Dict[str, str] = {}
    for need in ("title", "abstract", "pdf_url"):
        match = [c for c in df.columns if c.lower() == need]
        if not match:
            sys.stderr.write(f"[error] Missing required column: {need}\n")
            sys.exit(1)
        colmap[need] = match[0]

    total_rows = len(df)

    mask = df[colmap["title"]].fillna("").map(contains_multimodal) | df[colmap["abstract"]].fillna("").map(contains_multimodal)
    matched = int(mask.sum())
    selected = df[mask].copy()

    rep_base = os.path.splitext(os.path.basename(args.csv))[0]
    rep_path = os.path.join(args.outdir, f"{rep_base}_multimodal_download_report.csv")
    report_rows: list[dict] = []
    for idx, row in df.iterrows():
        report_rows.append({
            "row_index": idx,
            "title": normalize_text(row[colmap["title"]]),
            "pdf_url": normalize_text(row[colmap["pdf_url"]]),
            "matched_multimodal": bool(mask.iloc[idx]),
            "download_attempted": False,
            "downloaded": False,
            "failure_reason": "",
            "attempts": 0,
            "saved_filename": "",
            "saved_path": "",
        })

    downloaded = 0
    failures = 0

    if selected.empty:
        print("[info] No rows matched multimodal patterns.")
        return

    print(f"[info] Matched {len(selected)} rows. Downloading to: {args.outdir}")
    for i, (idx, row) in enumerate(tqdm(selected.iterrows(), total=len(selected), desc="Downloading")):
        pdf_url = str(row[colmap["pdf_url"]]).strip()
        if not pdf_url or pdf_url.lower() == "nan":
            continue

        title_from_csv = str(row[colmap["title"]]).strip()
        safe_title = sanitize_title_for_filename(title_from_csv) if title_from_csv else "untitled"

        final_name = f"([C]-{args.year}-{args.conf}){safe_title}.pdf"
        outpath = os.path.join(args.outdir, final_name)

        report_rows[idx]["download_attempted"] = True
        report_rows[idx]["saved_filename"] = final_name
        report_rows[idx]["saved_path"] = outpath

        if args.dry_run:
            print(f"[dry-run] {pdf_url}  ->  {outpath}")
            continue

        ok, reason, attempts = download_pdf(pdf_url, outpath, overwrite=args.overwrite)
        report_rows[idx]["attempts"] = attempts
        if ok:
            downloaded += 1
            report_rows[idx]["downloaded"] = True
            report_rows[idx]["failure_reason"] = ""
        else:
            failures += 1
            report_rows[idx]["downloaded"] = False
            report_rows[idx]["failure_reason"] = reason or "download_failed"

    report_df = pd.DataFrame(report_rows, columns=[
        "row_index",
        "title",
        "pdf_url",
        "matched_multimodal",
        "download_attempted",
        "downloaded",
        "failure_reason",
        "attempts",
        "saved_filename",
        "saved_path",
    ])
    report_df.to_csv(rep_path, index=False, encoding="utf-8")
    print()
    print("=== Summary ===")
    print(f"Total rows: {total_rows}")
    print(f"Matched: {matched}")
    print(f"Downloaded: {downloaded}")
    print(f"Failed: {failures}")
    print(f"Report saved: {rep_path}")


if __name__ == "__main__":
    main()
