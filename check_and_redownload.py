#!/usr/bin/env python3
"""
Check downloaded GDC files for given TCGA projects, validate integrity,
and re-download missing or corrupted files.

Usage:
    python check_and_redownload.py --projects TCGA-LUAD TCGA-PRAD
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict

import requests

# Load settings lazily
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import GDC_ENDPOINT, RAW_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def list_gdc_files(project_id: str, size: int = 2000) -> List[Dict]:
    url = f"{GDC_ENDPOINT}/files"
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id",
        "format": "JSON",
        "size": str(size),
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]
    return hits


def check_file(path: Path) -> bool:
    """Basic integrity checks: exists, non-zero, readable TSV header."""
    try:
        if not path.exists():
            return False
        if path.stat().st_size == 0:
            return False
        # Try reading first few bytes and first line
        with path.open("rb") as fh:
            start = fh.read(512)
            # If gzipped, try decompress
            if start[:2] == b"\x1f\x8b":
                import gzip
                fh.seek(0)
                with gzip.GzipFile(fileobj=fh) as gz:
                    head = gz.read(1024).decode("utf-8", errors="ignore")
            else:
                fh.seek(0)
                head = fh.read(1024).decode("utf-8", errors="ignore")
        lines = [l for l in head.splitlines() if l.strip()]
        if not lines:
            return False
        # basic TSV check: line contains at least 2 columns separated by tab
        if "\t" not in lines[0]:
            return False
        return True
    except Exception as e:
        logger.debug("Integrity check failed for %s: %s", path, e)
        return False


def download_file(file_id: str, dest: Path, max_attempts: int = 10, timeout: int = 120, backoff_factor: float = 1.0) -> bool:
    """Robust streamed download with retries, exponential backoff and temp-file writes."""
    url = f"{GDC_ENDPOINT}/data/{file_id}"
    session = requests.Session()
    try:
        # Use urllib3 Retry for idempotent error handling at adapter level
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter

        retries = Retry(
            total=max_attempts,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
    except Exception:
        # If urllib3 not available or fails, fall back to plain session
        pass

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            with session.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp_path = dest.with_suffix(dest.suffix + ".part")
                with tmp_path.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                tmp_path.replace(dest)
                return True
        except Exception as e:
            logger.warning("Download attempt %d/%d failed for %s: %s", attempt, max_attempts, file_id, e)
            # exponential backoff with jitter
            sleep_for = backoff_factor * (2 ** (attempt - 1))
            sleep_for = sleep_for + (0.1 * sleep_for)
            time.sleep(min(sleep_for, 600))
    return False


def process_project(project_id: str, max_retries: int = 3, timeout: int = 180, backoff_factor: float = 1.0) -> Dict:
    raw_dir = RAW_DIR / project_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_dir = RESULTS_DIR / project_id
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Querying GDC for project %s", project_id)
    hits = list_gdc_files(project_id)
    logger.info("GDC reports %d files for %s", len(hits), project_id)

    report = {"project": project_id, "total_expected": len(hits), "checked": 0, "ok": [], "missing": [], "corrupt": [], "redownloaded": []}

    for h in hits:
        fname = h.get("file_name")
        fid = h.get("file_id")
        if not fname or not fid:
            continue
        local_path = raw_dir / fname
        ok = check_file(local_path)
        report["checked"] += 1
        if ok:
            report["ok"].append(fname)
            continue
        # Missing or corrupt: try to re-download
        if not local_path.parent.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Re-downloading %s (%s)", fname, fid)
            success = download_file(fid, local_path, max_attempts=max_retries, timeout=timeout, backoff_factor=backoff_factor)
        if success and check_file(local_path):
            report["redownloaded"].append(fname)
        else:
            if local_path.exists() and local_path.stat().st_size == 0:
                local_path.unlink(missing_ok=True)
            if local_path.exists():
                report["corrupt"].append(fname)
            else:
                report["missing"].append(fname)

    # persist report
    out = results_dir / "download_check.json"
    out.write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s", out)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", nargs="+", required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=20, help="Maximum download attempts per file (exponential backoff)")
    parser.add_argument("--timeout", type=int, default=180, help="Per-request timeout in seconds")
    parser.add_argument("--backoff-factor", type=float, default=1.0, help="Backoff factor for exponential backoff")
    args = parser.parse_args()

    overall = {}
    for p in args.projects:
        try:
            r = process_project(p, max_retries=args.retries)
            # After initial pass, try re-downloading remaining with stronger attempts
            if (r.get("missing") or r.get("corrupt")):
                logger.info("Found %d missing + %d corrupt files, running stronger re-downloads...",
                            len(r.get("missing", [])), len(r.get("corrupt", [])))
                # attempt more persistent re-downloads for the problematic files
                raw_dir = RAW_DIR / p
                hits = list_gdc_files(p)
                for h in hits:
                    fname = h.get("file_name")
                    fid = h.get("file_id")
                    if not fname or not fid:
                        continue
                    local_path = raw_dir / fname
                    if check_file(local_path):
                        continue
                    logger.info("Persistent re-download for %s (%s)", fname, fid)
                    download_file(fid, local_path, max_attempts=args.max_attempts, timeout=args.timeout, backoff_factor=args.backoff_factor)
                # re-generate report
                r = process_project(p, max_retries=args.retries, timeout=args.timeout, backoff_factor=args.backoff_factor)
            overall[p] = r
        except Exception as e:
            logger.error("Failed processing %s: %s", p, e)
            overall[p] = {"error": str(e)}

    summary = Path("results") / "download_checks_summary.json"
    summary.write_text(json.dumps(overall, indent=2))
    logger.info("Summary written to %s", summary)

    # print brief summary
    for p, r in overall.items():
        if "error" in r:
            logger.error("%s: ERROR %s", p, r["error"])
        else:
            logger.info("%s: expected=%d checked=%d ok=%d redownloaded=%d missing=%d corrupt=%d",
                        p, r["total_expected"], r["checked"], len(r["ok"]), len(r["redownloaded"]), len(r["missing"]), len(r["corrupt"]))
