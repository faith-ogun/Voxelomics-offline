"""
Voxelomics MDT Command Service - Tools

Tooling layer for:
- Local evidence lookup for offline mode
- Structured output parsing helpers
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import httpx

from env_loader import load_service_env
from models import (
    EvidenceItem,
    MDTCaseInput,
)

logger = logging.getLogger(__name__)

# Load `.env` / `.env.local` for this service before reading os.environ.
load_service_env()

BASE_DIR = Path(__file__).parent
MOCK_DB_DIR = BASE_DIR / "mock_db"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _safe_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    if not match:
        return None
    try:
        return int(match.group(0))
    except (TypeError, ValueError):
        return None


@dataclass
class _CacheEntry:
    expires_at: float
    value: Any


class TTLCache:
    """
    Lightweight in-memory TTL cache for retrieval results.
    """

    def __init__(self, ttl_seconds: int, max_entries: int = 256) -> None:
        self.ttl_seconds = max(1, ttl_seconds)
        self.max_entries = max(16, max_entries)
        self._store: Dict[str, _CacheEntry] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            if entry.expires_at <= now:
                self._store.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: Any) -> None:
        now = time.time()
        with self._lock:
            if len(self._store) >= self.max_entries:
                oldest_key = min(self._store, key=lambda k: self._store[k].expires_at)
                self._store.pop(oldest_key, None)
            self._store[key] = _CacheEntry(expires_at=now + self.ttl_seconds, value=value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


def _load_json(file_name: str) -> Dict[str, Any]:
    path = MOCK_DB_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Mock data file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_local_db_path(env_name: str, default_file_name: str) -> Path:
    explicit_path = (os.getenv(env_name) or "").strip()
    if explicit_path:
        path = Path(explicit_path).expanduser()
    else:
        evidence_dir = (os.getenv("MDT_LOCAL_EVIDENCE_DIR") or "").strip()
        if evidence_dir:
            path = Path(evidence_dir).expanduser() / default_file_name
        else:
            path = (BASE_DIR / "local_data" / "evidence_cache" / default_file_name).resolve()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


def _load_json_from_path(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Local data file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_to_path(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_available_case_ids() -> List[str]:
    data = _load_json("cases.json")
    return [case["case_id"] for case in data.get("cases", [])]


def get_case_input(case_id: str) -> MDTCaseInput:
    data = _load_json("cases.json")
    for case in data.get("cases", []):
        if case.get("case_id") == case_id:
            return MDTCaseInput.model_validate(case)
    raise KeyError(f"Case not found: {case_id}")


def extract_gene_symbols(case_input: MDTCaseInput) -> List[str]:
    genes = {m.gene.upper() for m in case_input.genomics.mutations if m.gene}
    if not genes:
        # fallback if mutations list is empty
        candidates = re.findall(r"\b[A-Z0-9]{3,8}\b", case_input.genomics.report_summary)
        genes.update(candidates[:5])
    return sorted(genes)


def _search_literature_evidence_mock(
    diagnosis: str,
    genes: List[str],
    max_results: int = 3,
) -> List[EvidenceItem]:
    data = _load_json("literature_evidence.json")
    return _search_literature_evidence_from_data(
        diagnosis=diagnosis,
        genes=genes,
        data=data,
        max_results=max_results,
    )


def _search_literature_evidence_local(
    diagnosis: str,
    genes: List[str],
    max_results: int = 3,
) -> List[EvidenceItem]:
    local_path = _resolve_local_db_path("MDT_LOCAL_LITERATURE_DB", "literature_evidence.json")
    data = _load_json_from_path(local_path)
    return _search_literature_evidence_from_data(
        diagnosis=diagnosis,
        genes=genes,
        data=data,
        max_results=max_results,
    )


def _search_literature_evidence_from_data(
    diagnosis: str,
    genes: List[str],
    data: Dict[str, Any],
    max_results: int = 3,
) -> List[EvidenceItem]:
    """
    Local evidence lookup. Filters by diagnosis keyword and gene overlap.
    """
    diagnosis_lower = diagnosis.lower()
    gene_set = {g.upper() for g in genes}

    scored: List[tuple[int, Dict[str, Any]]] = []
    for item in data.get("evidence", []):
        score = 0
        diagnosis_tags = [d.lower() for d in item.get("diagnosis_tags", [])]
        gene_tags = {g.upper() for g in item.get("gene_tags", [])}

        if any(tag in diagnosis_lower for tag in diagnosis_tags):
            score += 2
        overlap = gene_set.intersection(gene_tags)
        score += len(overlap)

        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [EvidenceItem.model_validate(item) for _, item in scored[:max_results]]

    return results


class LiveRetrievalClient:
    """
    Live adapters for PubMed with TTL caching.
    """

    def __init__(self) -> None:
        self.timeout_seconds = max(1.0, _env_float("MDT_RETRIEVAL_TIMEOUT_SECONDS", 8.0))
        self.cache = TTLCache(
            ttl_seconds=max(30, _env_int("MDT_RETRIEVAL_CACHE_TTL_SECONDS", 3600)),
            max_entries=max(32, _env_int("MDT_RETRIEVAL_CACHE_MAX_ENTRIES", 512)),
        )
        self.pubmed_email = os.getenv("MDT_PUBMED_EMAIL")
        self.pubmed_tool = os.getenv("MDT_PUBMED_TOOL", "voxelomics-mdt-command")
        self.pubmed_api_key = os.getenv("MDT_PUBMED_API_KEY")

    def _cache_key(self, namespace: str, diagnosis: str, genes: List[str], max_results: int) -> str:
        normalized_genes = ",".join(sorted({g.upper() for g in genes}))
        return f"{namespace}|{diagnosis.strip().lower()}|{normalized_genes}|{max_results}"

    def fetch_pubmed_evidence(
        self,
        diagnosis: str,
        genes: List[str],
        max_results: int = 3,
    ) -> List[EvidenceItem]:
        key = self._cache_key("pubmed", diagnosis, genes, max_results)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        term_parts = [f'("{diagnosis}"[Title/Abstract])']
        if genes:
            gene_clause = " OR ".join([f"{g}[Title/Abstract]" for g in genes[:6]])
            term_parts.append(f"({gene_clause})")
        search_term = " AND ".join(term_parts)

        params: Dict[str, Any] = {
            "db": "pubmed",
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance",
            "term": search_term,
            "tool": self.pubmed_tool,
        }
        if self.pubmed_email:
            params["email"] = self.pubmed_email
        if self.pubmed_api_key:
            params["api_key"] = self.pubmed_api_key

        with httpx.Client(timeout=self.timeout_seconds) as client:
            search_resp = client.get(PUBMED_SEARCH_URL, params=params)
            search_resp.raise_for_status()
            search_payload = search_resp.json()

            id_list = (
                search_payload.get("esearchresult", {}).get("idlist", [])
                if isinstance(search_payload, dict)
                else []
            )
            if not id_list:
                self.cache.set(key, [])
                return []

            summary_params: Dict[str, Any] = {
                "db": "pubmed",
                "retmode": "json",
                "id": ",".join(id_list),
                "tool": self.pubmed_tool,
            }
            if self.pubmed_email:
                summary_params["email"] = self.pubmed_email
            if self.pubmed_api_key:
                summary_params["api_key"] = self.pubmed_api_key

            summary_resp = client.get(PUBMED_SUMMARY_URL, params=summary_params)
            summary_resp.raise_for_status()
            summary_payload = summary_resp.json()

        result_root = summary_payload.get("result", {}) if isinstance(summary_payload, dict) else {}
        uids = result_root.get("uids", [])
        evidence: List[EvidenceItem] = []
        for uid in uids[:max_results]:
            item = result_root.get(uid, {})
            if not isinstance(item, dict):
                continue
            title = item.get("title") or "PubMed article"
            source = item.get("fulljournalname") or item.get("source") or "PubMed"
            pubdate = item.get("pubdate") or item.get("sortpubdate")
            year = _safe_year(pubdate)
            finding = (
                f"Matched article for {diagnosis}; relevance inferred from title/abstract query."
            )
            evidence.append(
                EvidenceItem(
                    title=str(title),
                    source=str(source),
                    year=year,
                    identifier=f"PMID:{uid}",
                    finding=finding,
                )
            )

        self.cache.set(key, evidence)
        return evidence

_LIVE_RETRIEVAL = LiveRetrievalClient()


def _retrieval_mode() -> str:
    mode = (os.getenv("MDT_RETRIEVAL_MODE", "local") or "local").strip().lower()
    if mode != "local":
        raise ValueError("Offline build supports only MDT_RETRIEVAL_MODE=local.")
    return "local"


def search_literature_evidence(
    diagnosis: str,
    genes: List[str],
    max_results: int = 3,
) -> List[EvidenceItem]:
    _retrieval_mode()
    try:
        return _search_literature_evidence_local(diagnosis, genes, max_results)
    except Exception as exc:
        logger.warning("Local evidence lookup unavailable; returning empty evidence list: %s", exc)
        return []


def parse_json_payload(raw_text: str) -> Dict[str, Any]:
    """
    Parse JSON text safely, allowing fenced markdown wrappers.
    """
    cleaned = raw_text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def reset_retrieval_cache_for_tests() -> None:
    """
    Test utility to clear retrieval cache between tests.
    """
    _LIVE_RETRIEVAL.cache.clear()


def get_local_evidence_sync_status() -> Dict[str, Any]:
    """
    Returns sync metadata and local cache file stats.
    """
    literature_path = _resolve_local_db_path("MDT_LOCAL_LITERATURE_DB", "literature_evidence.json")
    status_path = _resolve_local_db_path("MDT_LOCAL_EVIDENCE_SYNC_STATUS_DB", "evidence_sync_status.json")

    literature_count = 0
    last_synced_at: Optional[str] = None

    try:
        if literature_path.exists():
            payload = _load_json_from_path(literature_path)
            literature_count = len(payload.get("evidence", []))
    except Exception:
        literature_count = 0

    try:
        if status_path.exists():
            payload = _load_json_from_path(status_path)
            raw = payload.get("last_synced_at")
            if isinstance(raw, str) and raw.strip():
                last_synced_at = raw.strip()
    except Exception:
        last_synced_at = None

    return {
        "literature_path": str(literature_path),
        "status_path": str(status_path),
        "literature_count": literature_count,
        "last_synced_at": last_synced_at,
    }


def sync_local_evidence_cache(
    diagnosis: str,
    genes: List[str],
    max_results: int = 8,
) -> Dict[str, Any]:
    """
    Pulls live PubMed results (when online) and persists a local snapshot
    used by offline retrieval.
    """
    clean_diagnosis = (diagnosis or "").strip()
    if not clean_diagnosis:
        raise ValueError("Diagnosis is required for evidence sync.")

    normalized_genes = sorted({g.strip().upper() for g in genes if str(g).strip()})
    max_results = max(1, min(20, int(max_results)))

    warnings: List[str] = []

    try:
        live_evidence = _LIVE_RETRIEVAL.fetch_pubmed_evidence(
            diagnosis=clean_diagnosis,
            genes=normalized_genes,
            max_results=max_results,
        )
    except Exception as exc:
        logger.warning("PubMed sync failed; continuing with available sources: %s", exc)
        live_evidence = []
        warnings.append(f"pubmed_failed:{exc}")

    literature_path = _resolve_local_db_path("MDT_LOCAL_LITERATURE_DB", "literature_evidence.json")
    status_path = _resolve_local_db_path("MDT_LOCAL_EVIDENCE_SYNC_STATUS_DB", "evidence_sync_status.json")

    existing_literature: Dict[str, Any] = {"evidence": []}
    if literature_path.exists():
        try:
            existing_literature = _load_json_from_path(literature_path)
        except Exception:
            existing_literature = {"evidence": []}

    lit_index: Dict[str, Dict[str, Any]] = {}
    for item in existing_literature.get("evidence", []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("identifier") or item.get("title") or "").strip()
        if not key:
            continue
        lit_index[key] = item

    for ev in live_evidence:
        key = (ev.identifier or ev.title).strip()
        diagnosis_tags = [clean_diagnosis.lower()]
        existing = lit_index.get(key, {})
        prior_diag = existing.get("diagnosis_tags", []) if isinstance(existing, dict) else []
        prior_genes = existing.get("gene_tags", []) if isinstance(existing, dict) else []
        lit_index[key] = {
            "title": ev.title,
            "source": ev.source,
            "year": ev.year,
            "identifier": ev.identifier,
            "finding": ev.finding,
            "diagnosis_tags": sorted(
                {str(x).strip().lower() for x in [*prior_diag, *diagnosis_tags] if str(x).strip()}
            ),
            "gene_tags": sorted(
                {str(x).strip().upper() for x in [*prior_genes, *normalized_genes] if str(x).strip()}
            ),
        }

    literature_payload = {"evidence": list(lit_index.values())}
    _write_json_to_path(literature_path, literature_payload)

    status_payload = {
        "last_synced_at": _iso_utc_now(),
        "query": {
            "diagnosis": clean_diagnosis,
            "genes": normalized_genes,
            "max_results": max_results,
        },
        "literature_count": len(literature_payload["evidence"]),
    }
    _write_json_to_path(status_path, status_payload)

    return {
        "diagnosis": clean_diagnosis,
        "genes": normalized_genes,
        "literature_count": len(literature_payload["evidence"]),
        "literature_path": str(literature_path),
        "last_synced_at": status_payload["last_synced_at"],
        "warnings": warnings,
    }
