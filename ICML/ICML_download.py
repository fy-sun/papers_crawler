# -*- coding: utf-8 -*-
"""
Read a CSV with columns: type, name, virtualsite_url, speakers/authors, abstract
1) Keep only rows where type == 'Poster' (case-insensitive, trimmed).
2) Check abstract for multimodal-related phrases (multimodal/multi-modal/cross-modal,
   vision-language/image-text/etc.), allowing punctuation around and Unicode dashes.
   (Regex/cleaning logic aligned with code-one style: clean_text + normalize_space + IGNORECASE.)
3) For matched rows, fetch virtualsite_url HTML, locate the unique anchor with title="OpenReview",
   extract the id from its href (e.g., .../forum?id=vlg5WRKHxh), construct
   https://openreview.net/pdf?id=<ID>, and download the PDF.

Changes made:
- Unify text cleaning & regex style with code-one (normalize_space/clean_text, IGNORECASE flags).
- Add 3-attempt retry via requests' Retry(total=3).
- Rename PDFs using code-one-like safe filename + required pattern:
  '([C]-2025-ICML)<title_or_id>.pdf'
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Poster-OpenReview-Downloader/1.0; +https://example.org/academic-use)",
    "Accept-Language": "en-US,en;q=0.8",
}

OPENREVIEW_PDF_PREFIX = "https://openreview.net/pdf?id="


class ScrapeError(Exception):
    """自定义抓取异常。"""


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def clean_text(s: str) -> str:
    return normalize_space(html.unescape(s or ""))


def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/*?"<>|:]+', "_", name or "")
    return name[:200].strip("_")


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

MM_RE = MULTIMODAL_REGEX_1


@dataclass
class PosterItem:
    row_index: int
    name: str
    virtualsite_url: str
    abstract: str


def build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=16, pool_maxsize=16)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    return s


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    retry=retry_if_exception_type((requests.RequestException, ScrapeError)),
)
def http_get(session: requests.Session, url: str, *, timeout: int = 30) -> requests.Response:
    r = session.get(url, timeout=timeout)
    if r.status_code >= 400:
        raise ScrapeError(f"HTTP {r.status_code} for {url}")
    return r


def extract_openreview_id_from_html(html_text: str) -> Optional[str]:
    """
    定位 <a title="OpenReview" href=".../forum?id=XYZ"> 并提取 id。
    """
    soup = BeautifulSoup(html_text, "html.parser")
    a = soup.find("a", attrs={"title": "OpenReview"}, href=True)
    if not a:
        return None
    href = (a.get("href") or "").strip()
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    id_vals = qs.get("id") or qs.get("Id") or qs.get("ID")
    if id_vals:
        return id_vals[0]
    m = re.search(r"[?&]id=([A-Za-z0-9_\-]+)", href)
    if m:
        return m.group(1)
    return None


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def download_pdf(session: requests.Session, pdf_url: str, outdir: str, fname: str) -> str:
    ensure_dir(outdir)
    path = os.path.join(outdir, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    with session.get(pdf_url, stream=True, timeout=30) as r:
        r.raise_for_status()
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, path)
    return path


def read_poster_items_from_csv(csv_path: str, encoding: str, delimiter: str) -> List[PosterItem]:
    """
    读取 CSV 并过滤：
    - 仅保留 type == 'Poster'（不区分大小写，前后空白已清洗）
    - 对 abstract 做清洗（clean_text），后续再进行多模态匹配
    返回 PosterItem 列表。
    """
    required = {"type", "name", "virtualsite_url", "abstract"}
    items: List[PosterItem] = []
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f"CSV 必须包含列：{sorted(required)}；实际为：{reader.fieldnames}")
        for i, row in enumerate(reader):
            t = clean_text(row.get("type") or "").lower()
            if t != "poster":
                continue
            name = clean_text(row.get("name") or "")
            url = (row.get("virtualsite_url") or "").strip()
            abstract = clean_text(row.get("abstract") or "")
            items.append(PosterItem(row_index=i, name=name, virtualsite_url=url, abstract=abstract))
    return items


def build_pdf_filename(name: str, openreview_id: Optional[str], year: int, venue: str, conf_tag: str = "C") -> str:
    suffix = name or openreview_id or "paper"
    suffix = safe_filename(suffix)
    return f"([{conf_tag}]-{year}-{venue}){suffix}.pdf"


def run(csv_path: str, outdir: str, encoding: str, delimiter: str, year: int, venue: str) -> None:
    session = build_session()

    posters = read_poster_items_from_csv(csv_path, encoding, delimiter)
    total_rows = 0
    total_posters = len(posters)
    matched = 0
    downloaded = 0
    failures = 0

    report_rows = []
    rep_path = os.path.join(outdir, "openreview_download_report.csv")

    print(f"[info] CSV: {csv_path}")
    print(f"[info] Found {total_posters} Poster rows after filtering type == 'Poster'")

    for item in tqdm(posters, desc="Processing posters"):
        total_rows += 1

        if not MM_RE.search(item.abstract or ""):
            report_rows.append({
                "row_index": item.row_index,
                "type": "Poster",
                "name": item.name,
                "virtualsite_url": item.virtualsite_url,
                "matched_multimodal": False,
                "openreview_id": "",
                "pdf_url": "",
                "saved_to": "",
                "error": "not multimodal",
            })
            continue

        matched += 1

        try:
            pr = http_get(session, item.virtualsite_url)
            or_id = extract_openreview_id_from_html(pr.text)
            if not or_id:
                failures += 1
                report_rows.append({
                    "row_index": item.row_index,
                    "type": "Poster",
                    "name": item.name,
                    "virtualsite_url": item.virtualsite_url,
                    "matched_multimodal": True,
                    "openreview_id": "",
                    "pdf_url": "",
                    "saved_to": "",
                    "error": "OpenReview id not found",
                })
                continue

            pdf_url = OPENREVIEW_PDF_PREFIX + or_id
            fname = build_pdf_filename(item.name, or_id, year=year, venue=venue)
            saved_path = download_pdf(session, pdf_url, outdir, fname)
            downloaded += 1

            report_rows.append({
                "row_index": item.row_index,
                "type": "Poster",
                "name": item.name,
                "virtualsite_url": item.virtualsite_url,
                "matched_multimodal": True,
                "openreview_id": or_id,
                "pdf_url": pdf_url,
                "saved_to": saved_path,
                "error": "",
            })

        except Exception as e:
            failures += 1
            report_rows.append({
                "row_index": item.row_index,
                "type": "Poster",
                "name": item.name,
                "virtualsite_url": item.virtualsite_url,
                "matched_multimodal": True,
                "openreview_id": "",
                "pdf_url": "",
                "saved_to": "",
                "error": f"{type(e).__name__}: {e}",
            })

    try:
        ensure_dir(outdir)
        cols = ["row_index", "type", "name", "virtualsite_url", "matched_multimodal",
                "openreview_id", "pdf_url", "saved_to", "error"]
        with open(rep_path, "w", encoding="utf-8", newline="") as rf:
            writer = csv.DictWriter(rf, fieldnames=cols)
            writer.writeheader()
            for r in report_rows:
                writer.writerow(r)
        print(f"[ok] Report saved: {rep_path}")
    except Exception as e:
        print(f"[WARN] Failed to write report CSV: {type(e).__name__}: {e}")

    print("=== Summary ===")
    print(f"Total Poster rows processed: {total_rows}")
    print(f"Matched multimodal: {matched}")
    print(f"Downloaded: {downloaded}")
    print(f"Failed: {failures}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download OpenReview PDFs for Poster entries with multimodal abstracts (code-one style).")
    p.add_argument("--csv", type=str, default="./files/ICML 2025 Events.csv",
                   help="Path to CSV with columns: type,name,virtualsite_url,speakers/authors,abstract")
    p.add_argument("--outdir", type=str, default="./downloads", help="Directory to save PDFs")
    p.add_argument("--encoding", type=str, default="utf-8", help="CSV encoding (e.g., utf-8 or utf-8-sig)")
    p.add_argument("--delimiter", type=str, default=",", help="CSV delimiter (default: ,)")
    p.add_argument("--year", type=int, default=2025, help="Year for filename prefix, e.g., 2025")
    p.add_argument("--conf", type=str, default="ICML", help="Venue tag for filename prefix, e.g., ICML")
    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    try:
        run(
            csv_path=args.csv,
            outdir=args.outdir,
            encoding=args.encoding,
            delimiter=args.delimiter,
            year=args.year,
            venue=args.conf,
        )
    except FileNotFoundError:
        print(f"[FATAL] CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
