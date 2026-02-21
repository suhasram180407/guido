"""
Step 5 (Part 1) – PubMed Literature Retrieval
===============================================
Uses Biopython's Entrez module to query PubMed and retrieve up to
PUBMED_MAX_ABSTRACTS abstracts per (gene, disease) keyword pair.

Abstracts are cached locally in JSON format to avoid repeated API calls.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from Bio import Entrez, Medline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    ENTREZ_API_KEY,
    ENTREZ_EMAIL,
    PUBMED_MAX_ABSTRACTS,
    RESULTS_DIR,
    TCGA_PROJECT_ID,
)

logger = logging.getLogger(__name__)

Entrez.email = ENTREZ_EMAIL
if ENTREZ_API_KEY:
    Entrez.api_key = ENTREZ_API_KEY

CACHE_DIR = RESULTS_DIR / "pubmed_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Disease keyword inferred from project ID (can be overridden)
_PROJECT_DISEASE_MAP = {
    "TCGA-BRCA": "breast cancer",
    "TCGA-LUAD": "lung adenocarcinoma",
    "TCGA-COAD": "colorectal cancer",
    "TCGA-GBM":  "glioblastoma",
    "TCGA-OV":   "ovarian cancer",
}


def get_disease_keyword(project_id: str = TCGA_PROJECT_ID) -> str:
    return _PROJECT_DISEASE_MAP.get(project_id, "cancer")


def _cache_file(gene: str, disease: str) -> Path:
    safe_gene = gene.replace("/", "_")
    safe_disease = disease.replace(" ", "_")
    return CACHE_DIR / f"{safe_gene}_{safe_disease}.json"


def fetch_pubmed_abstracts(
    gene: str,
    disease_keyword: str,
    max_results: int = PUBMED_MAX_ABSTRACTS,
) -> List[Dict[str, str]]:
    """
    Search PubMed for `gene disease_keyword` and return up to max_results
    structured abstract records.

    Each record contains:
        pmid, title, abstract, authors, journal, year

    Results are cached locally to disk.
    """
    cache = _cache_file(gene, disease_keyword)
    if cache.exists():
        logger.debug("Cache hit: %s", cache)
        return json.loads(cache.read_text(encoding="utf-8"))

    query = f"{gene}[Gene Name] AND {disease_keyword}[Title/Abstract]"
    try:
        # Search
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        pmids: List[str] = record.get("IdList", [])

        if not pmids:
            logger.info("No PubMed results for gene=%s, disease=%s", gene, disease_keyword)
            cache.write_text(json.dumps([]), encoding="utf-8")
            return []

        time.sleep(0.35)  # NCBI rate limit: 3 requests/sec without API key

        # Fetch record details
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="text")
        records = list(Medline.parse(handle))
        handle.close()

        abstracts = []
        for rec in records:
            abstracts.append({
                "pmid":     rec.get("PMID", ""),
                "title":    rec.get("TI",   ""),
                "abstract": rec.get("AB",   ""),
                "authors":  ", ".join(rec.get("AU", [])),
                "journal":  rec.get("TA",   ""),
                "year":     rec.get("DP",   "")[:4] if rec.get("DP") else "",
            })

        cache.write_text(json.dumps(abstracts, ensure_ascii=False), encoding="utf-8")
        logger.info("Fetched %d abstracts for gene=%s", len(abstracts), gene)
        return abstracts

    except Exception as exc:
        logger.warning("PubMed fetch failed for gene=%s: %s", gene, exc)
        return []


def fetch_all_biomarkers(
    gene_list: List[str],
    disease_keyword: Optional[str] = None,
    project_id: str = TCGA_PROJECT_ID,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Retrieve PubMed abstracts for all genes in gene_list.

    Returns
    -------
    {gene_id: [abstract_records]}
    """
    if disease_keyword is None:
        disease_keyword = get_disease_keyword(project_id)

    results: Dict[str, List[Dict[str, str]]] = {}
    for gene in gene_list:
        results[gene] = fetch_pubmed_abstracts(gene, disease_keyword)
        time.sleep(0.1)  # gentle throttle

    return results
