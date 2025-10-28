#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cvpr_scraper.py

从 CVF Open Access 抓取指定年份的 CVPR 论文，输出 CSV 与 .bib（向官方 bib 注入摘要）。
示例：
    python cvpr_scraper.py --year 2025 --outdir ./cvpr2025
    python cvpr_scraper.py --year 2024 --outdir ./cvpr2024 --csv-full

注意：
- 仅供学术检索与个人研究使用，请遵守目标网站的使用条款与爬取礼仪。
- 默认速率较温和（每篇短暂停顿），可按需在 polite_sleep 中调整。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm


BASE = "https://openaccess.thecvf.com/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CVPR-Scraper/1.0; +https://example.org/academic-use)",
    "Accept-Language": "en-US,en;q=0.8",
}


class ScrapeError(Exception):
    """自定义抓取异常。"""


@dataclass
class ListingItem:
    paper_url: str
    pdf_url: Optional[str]
    title_from_dt: str
    bib_text: Optional[str]


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def clean_text(s: str) -> str:
    return normalize_space(html.unescape(s or ""))


def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/*?"<>|:]+', "_", name or "")
    return name[:200].strip("_")


def split_authors_bibtex(author_field: str) -> List[str]:
    authors = [normalize_space(a) for a in re.split(r"\s+and\s+", author_field or "")]
    authors = [a.rstrip(",") for a in authors if a]
    return authors


def guess_bib_key_from_bib(bib_text: str) -> Optional[str]:
    m = re.search(r"@\w+\s*\{\s*([^,\s]+)", bib_text or "")
    return m.group(1) if m else None


def inject_abstract_into_bib(bib_text: str, abstract: str) -> str:
    abstract = (abstract or "").replace("\n", " ").strip()
    bib_wo_abs = re.sub(
        r"\n\s*abstract\s*=\s*\{.*?\},?\s*\n",
        "\n",
        bib_text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )

    insert_pos = bib_wo_abs.rfind("}")
    if insert_pos == -1:
        return bib_text or ""  # 回退

    snippet = ',\n  abstract={' + abstract.replace("}", r"\}") + "}\n"
    return bib_wo_abs[:insert_pos] + snippet + bib_wo_abs[insert_pos:]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    retry=retry_if_exception_type((requests.RequestException, ScrapeError)),
)
def http_get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        raise ScrapeError(f"HTTP {r.status_code} for {url}")
    return r


def find_listing_url(year: int) -> str:
    """
    入口优先尝试：
      1) https://openaccess.thecvf.com/CVPR{year}?day=all
      2) https://openaccess.thecvf.com/CVPR{year}
    """
    candidates = [
        f"{BASE}CVPR{year}?day=all",
        f"{BASE}CVPR{year}",
    ]
    for u in candidates:
        try:
            r = http_get(u)
            if "ptitle" in r.text or "bibref" in r.text or "papers" in r.text:
                return u
        except Exception:
            continue
    raise RuntimeError(f"未找到年份 {year} 的总表入口。尝试过：{candidates}")


def parse_listing_items_with_bib(html_text: str) -> List[ListingItem]:
    """
    从 CVPR?day=all 总表页提取每篇的：
      - 详情页链接 paper_url
      - PDF 链接 pdf_url（若有）
      - 标题（来自 dt.ptitle > a）
      - 官方 bib 文本（来自 div.bibref）
    说明：总表结构通常是 dt.ptitle 后面跟着多个 dd，其中一个 dd 里包含 link2/bibref。
    """
    soup = BeautifulSoup(html_text, "html.parser")
    items: List[ListingItem] = []

    for dt in soup.select("dt.ptitle"):
        a = dt.find("a")
        if not a or not a.get("href"):
            continue
        paper_url = urljoin(BASE, a["href"])
        title_from_dt = clean_text(a.get_text(" "))

        pdf_url: Optional[str] = None
        bib_text: Optional[str] = None

        node = dt.find_next_sibling()
        while node and node.name == "dd":
            if pdf_url is None:
                pdf_a = node.find("a", string=lambda s: s and "pdf" in s.lower())
                if pdf_a and pdf_a.get("href"):
                    pdf_url = urljoin(BASE, pdf_a["href"])

            if bib_text is None:
                bib_div = node.find("div", class_=lambda c: c and "bibref" in c)
                if bib_div:
                    text = bib_div.get_text("\n")
                    bib_text = text.strip() if text else None

            nxt = node.find_next_sibling()
            if not nxt or nxt.name == "dt":
                break
            node = nxt

        items.append(ListingItem(paper_url=paper_url, pdf_url=pdf_url, title_from_dt=title_from_dt, bib_text=bib_text))

    uniq: List[ListingItem] = []
    seen = set()
    for it in items:
        if it.paper_url not in seen:
            uniq.append(it)
            seen.add(it.paper_url)
    return uniq


def extract_abstract_from_paper(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")

    node = soup.find(id="abstract") or soup.find("div", class_="abstract")
    if node:
        return clean_text(node.get_text(" "))

    h2 = soup.find(["h2", "h3"], string=lambda s: s and "abstract" in s.lower())
    if h2:
        nxt = h2.find_next_sibling()
        if nxt:
            return clean_text(nxt.get_text(" "))

    return ""


def find_bib_url_from_paper(html_text: str, paper_url: str) -> Optional[str]:
    soup = BeautifulSoup(html_text, "html.parser")
    a = soup.find("a", string=lambda s: s and "bibtex" in s.lower())
    if a and a.get("href"):
        return urljoin(paper_url, a["href"])
    a2 = soup.find("a", href=lambda h: h and h.lower().endswith(".bib"))
    if a2 and a2.get("href"):
        return urljoin(paper_url, a2["href"])
    return None


def parse_bib_fields(bib_text: str) -> Dict[str, object]:
    fields: Dict[str, object] = {}
    for k in ["author", "title", "booktitle", "month", "year", "pages", "url"]:
        m = re.search(rf"{k}\s*=\s*\{{(.*?)\}}", bib_text or "", flags=re.IGNORECASE | re.DOTALL)
        if m:
            fields[k.lower()] = clean_text(m.group(1))

    if "author" in fields:
        fields["authors_list"] = split_authors_bibtex(str(fields["author"]))
    else:
        fields["authors_list"] = []

    return fields


def polite_sleep(i: int) -> None:
    if i and i % 20 == 0:
        time.sleep(3)
    else:
        time.sleep(0.3)


def build_csv_row_minimal(fields: Dict[str, object], abstract: str, paper_url: str) -> Dict[str, str]:
    title = str(fields.get("title", "")) if fields else ""
    authors_list = fields.get("authors_list", []) if fields else []
    authors = "; ".join([str(a) for a in authors_list])
    return {
        "title": title,
        "authors": authors,
        "url": paper_url,
        "abstract": abstract or "",
    }


def build_csv_row_full(
    bib_key: str,
    fields: Dict[str, object],
    abstract: str,
    paper_url: str,
    pdf_url: Optional[str],
) -> Dict[str, str]:
    return {
        "bibkey": bib_key,
        "title": str(fields.get("title", "")),
        "authors": "; ".join([str(a) for a in fields.get("authors_list", [])]),
        "booktitle": str(fields.get("booktitle", "CVPR")),
        "month": str(fields.get("month", "")),
        "year": str(fields.get("year", "")),
        "pages": str(fields.get("pages", "")),
        "url": str(fields.get("url", paper_url or "")),
        "pdf_url": pdf_url or "",
        "paper_page": paper_url or "",
        "abstract": abstract or "",
    }


def run(year: int, outdir: str, csv_name: Optional[str], bib_name: Optional[str], csv_full: bool) -> None:
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(outdir, csv_name or (f"cvpr{year}_full.csv" if csv_full else f"cvpr{year}.csv"))
    bib_path = os.path.join(outdir, bib_name or f"cvpr{year}.bib")

    listing_url = find_listing_url(year)
    resp = http_get(listing_url)

    items = parse_listing_items_with_bib(resp.text)
    if not items:
        raise RuntimeError("未在总表中解析到论文条目，页面结构可能变化。")

    print(f"[info] Year {year}: found {len(items)} papers from listing {listing_url}")

    csv_rows: List[Dict[str, str]] = []
    bib_items: List[str] = []

    for i, item in enumerate(tqdm(items, desc="Scraping papers")):
        paper_url = item.paper_url
        pdf_url = item.pdf_url
        official_bib = item.bib_text

        try:
            pr = http_get(paper_url)
            paper_html = pr.text
            abstract = extract_abstract_from_paper(paper_html)

            fields: Dict[str, object] = {}
            if official_bib:
                fields = parse_bib_fields(official_bib)
                bib_key = guess_bib_key_from_bib(official_bib)
                if not bib_key:
                    title_for_key = str(fields.get("title", "")) or item.title_from_dt
                    first_author = str(fields.get("authors_list", ["anon"])[0])
                    h = hashlib.md5((title_for_key + first_author).encode("utf-8")).hexdigest()[:8]
                    bib_key = f"CVPR{year}_{h}"

                merged_bib = inject_abstract_into_bib(official_bib, abstract)
                bib_items.append(merged_bib)

                if csv_full:
                    row = build_csv_row_full(bib_key=bib_key, fields=fields, abstract=abstract, paper_url=paper_url, pdf_url=pdf_url)
                else:
                    row = build_csv_row_minimal(fields=fields, abstract=abstract, paper_url=paper_url)
                csv_rows.append(row)

            else:
                bib_url = find_bib_url_from_paper(paper_html, paper_url)
                fetched_bib: Optional[str] = None
                if bib_url:
                    br = http_get(bib_url)
                    fetched_bib = br.text

                if fetched_bib:
                    fields = parse_bib_fields(fetched_bib)
                    bib_key = guess_bib_key_from_bib(fetched_bib) or f"CVPR{year}_" + hashlib.md5((paper_url).encode()).hexdigest()[:8]
                    merged_bib = inject_abstract_into_bib(fetched_bib, abstract)
                    bib_items.append(merged_bib)

                    if csv_full:
                        row = build_csv_row_full(bib_key=bib_key, fields=fields, abstract=abstract, paper_url=paper_url, pdf_url=pdf_url)
                    else:
                        row = build_csv_row_minimal(fields=fields, abstract=abstract, paper_url=paper_url)
                    csv_rows.append(row)

                else:
                    title = item.title_from_dt
                    bib_key = f"CVPR{year}:" + hashlib.md5((title + paper_url).encode("utf-8")).hexdigest()[:8]
                    min_bib = f"""@InProceedings{{{bib_key},
  author={{ }},
  title={{ {title} }},
  booktitle={{CVPR}},
  month={{}},
  year={{{year}}}
}}"""
                    bib_items.append(inject_abstract_into_bib(min_bib, abstract))

                    if csv_full:
                        row = {
                            "bibkey": bib_key,
                            "title": title,
                            "authors": "",
                            "booktitle": "CVPR",
                            "month": "",
                            "year": str(year),
                            "pages": "",
                            "url": paper_url,
                            "pdf_url": pdf_url or "",
                            "paper_page": paper_url,
                            "abstract": abstract or "",
                        }
                    else:
                        row = {"title": title, "authors": "", "url": paper_url, "abstract": abstract or ""}
                    csv_rows.append(row)

        except Exception as e:
            if csv_full:
                csv_rows.append(
                    {
                        "bibkey": f"ERROR_{i}",
                        "title": "",
                        "authors": "",
                        "booktitle": "CVPR",
                        "month": "",
                        "year": str(year),
                        "pages": "",
                        "url": paper_url,
                        "pdf_url": pdf_url or "",
                        "paper_page": paper_url,
                        "abstract": f"[ERROR] {e}",
                    }
                )
            else:
                csv_rows.append({"title": "", "authors": "", "url": paper_url, "abstract": f"[ERROR] {e}"})

        polite_sleep(i)

    if csv_full:
        df = pd.DataFrame(
            csv_rows,
            columns=[
                "bibkey",
                "title",
                "authors",
                "booktitle",
                "month",
                "year",
                "pages",
                "url",
                "pdf_url",
                "paper_page",
                "abstract",
            ],
        )
    else:
        df = pd.DataFrame(csv_rows, columns=["title", "authors", "url", "abstract"])

    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    print(f"[ok] CSV saved: {csv_path}")

    with open(bib_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(bib_items).strip() + "\n")
    print(f"[ok] BibTeX saved: {bib_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape CVPR proceedings (CVF Open Access) for a given year.")
    parser.add_argument("--year", type=int, default=2025, help="CVPR year, e.g., 2025")
    parser.add_argument("--outdir", type=str, default="./files", help="Output directory")
    parser.add_argument("--csv-name", type=str, default=None, help="CSV filename (optional)")
    parser.add_argument("--bib-name", type=str, default=None, help="BIB filename (optional)")
    parser.add_argument(
        "--csv-full",
        action="store_true",
        default=True,
        help="Write a full CSV (bibkey, title, authors, booktitle, month, year, pages, url, pdf_url, paper_page, abstract). "
        "If not set, only four columns are written: title, authors, url, abstract.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run(year=args.year, outdir=args.outdir, csv_name=args.csv_name, bib_name=args.bib_name, csv_full=args.csv_full)


if __name__ == "__main__":
    main()
