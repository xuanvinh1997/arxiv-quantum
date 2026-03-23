"""Tools for fetching papers from arXiv and Semantic Scholar."""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import feedparser
import httpx

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.toml"


def _load_config() -> dict:
    path = Path(os.environ.get("RESEARCH_CONFIG", _CONFIG_PATH))
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    def __init__(self, rps: float) -> None:
        self._min_interval = 1.0 / rps
        self._last_call = 0.0

    async def wait(self) -> None:
        now = time.monotonic()
        wait = self._min_interval - (now - self._last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    doi: str | None
    arxiv_id: str | None
    title: str
    abstract: str
    authors: list[str]
    published_date: str
    updated_date: str | None
    pdf_url: str | None
    categories: list[str]
    citation_count: int | None = None
    reference_count: int | None = None
    s2_paper_id: str | None = None
    embedding: list[float] | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ---------------------------------------------------------------------------
# arXiv client
# ---------------------------------------------------------------------------

class ArxivClient:
    def __init__(self, cfg: dict) -> None:
        self._base_url = cfg.get("base_url", "http://export.arxiv.org/api/query")
        rps = cfg.get("rate_limit_rps", 3)
        self._rl = _RateLimiter(rps)

    @classmethod
    def from_config(cls) -> "ArxivClient":
        return cls(_load_config()["arxiv"])

    async def search(
        self,
        query: str,
        max_results: int = 20,
        categories: list[str] | None = None,
        sort_by: str = "relevance",
    ) -> list[Paper]:
        """Search arXiv and return structured Paper objects."""
        await self._rl.wait()
        search_query = query
        if categories:
            cat_filter = " OR ".join(f"cat:{c}" for c in categories)
            search_query = f"({query}) AND ({cat_filter})"

        params = {
            "search_query": search_query,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(self._base_url, params=params)
            resp.raise_for_status()

        feed = feedparser.parse(resp.text)
        papers = []
        for entry in feed.entries:
            papers.append(self._parse_entry(entry))
        return papers

    async def get_paper(self, arxiv_id: str) -> Paper | None:
        """Fetch a single paper by arXiv ID."""
        # Strip version suffix for query
        base_id = re.sub(r"v\d+$", "", arxiv_id)
        await self._rl.wait()
        params = {"id_list": base_id, "max_results": 1}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(self._base_url, params=params)
            resp.raise_for_status()

        feed = feedparser.parse(resp.text)
        if not feed.entries:
            return None
        return self._parse_entry(feed.entries[0])

    @staticmethod
    def _parse_entry(entry: Any) -> Paper:
        arxiv_id = entry.get("id", "").split("/abs/")[-1].split("v")[0]
        doi = entry.get("arxiv_doi") or None

        # Build synthetic DOI from arXiv ID if no real DOI
        if not doi:
            doi = f"arxiv:{arxiv_id}"

        authors = [a.get("name", "") for a in entry.get("authors", [])]
        categories = [t.get("term", "") for t in entry.get("tags", [])]

        pdf_url = None
        for link in entry.get("links", []):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break

        published = entry.get("published", "")[:10]
        updated = entry.get("updated", "")[:10]

        return Paper(
            doi=doi,
            arxiv_id=arxiv_id,
            title=entry.get("title", "").replace("\n", " ").strip(),
            abstract=entry.get("summary", "").replace("\n", " ").strip(),
            authors=authors,
            published_date=published,
            updated_date=updated if updated != published else None,
            pdf_url=pdf_url,
            categories=categories,
        )


# ---------------------------------------------------------------------------
# Semantic Scholar client
# ---------------------------------------------------------------------------

_S2_PAPER_FIELDS = (
    "paperId,externalIds,title,abstract,authors,year,publicationDate,"
    "citationCount,referenceCount,fieldsOfStudy,openAccessPdf"
)
_S2_CITATION_FIELDS = "citingPaper.paperId,citingPaper.externalIds,citingPaper.title,citingPaper.abstract,citingPaper.authors,citingPaper.year,citingPaper.citationCount"
_S2_REFERENCE_FIELDS = "citedPaper.paperId,citedPaper.externalIds,citedPaper.title,citedPaper.abstract,citedPaper.authors,citedPaper.year,citedPaper.citationCount"


class SemanticScholarClient:
    def __init__(self, cfg: dict) -> None:
        self._base_url = cfg.get("base_url", "https://api.semanticscholar.org/graph/v1")
        api_key = cfg.get("api_key", "")
        self._headers = {"x-api-key": api_key} if api_key else {}
        # 1 req/s with key, 100 req/5min without
        rps = 1.0 if api_key else 0.33
        self._rl = _RateLimiter(rps)

    @classmethod
    def from_config(cls) -> "SemanticScholarClient":
        return cls(_load_config()["semantic_scholar"])

    async def get_paper(self, paper_id: str) -> dict | None:
        """Fetch paper metadata. paper_id can be S2 ID, DOI, or ArXiv:<id>."""
        await self._rl.wait()
        url = f"{self._base_url}/paper/{paper_id}"
        async with httpx.AsyncClient(timeout=30, headers=self._headers) as client:
            resp = await client.get(url, params={"fields": _S2_PAPER_FIELDS})
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()

    async def search(
        self,
        query: str,
        limit: int = 20,
        fields: str | None = None,
    ) -> list[dict]:
        """Search Semantic Scholar papers."""
        await self._rl.wait()
        url = f"{self._base_url}/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": fields or _S2_PAPER_FIELDS,
        }
        async with httpx.AsyncClient(timeout=30, headers=self._headers) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])

    async def get_citations(self, paper_id: str, limit: int = 50) -> list[dict]:
        """Get papers that cite the given paper."""
        await self._rl.wait()
        url = f"{self._base_url}/paper/{paper_id}/citations"
        params = {"fields": _S2_CITATION_FIELDS, "limit": min(limit, 1000)}
        async with httpx.AsyncClient(timeout=30, headers=self._headers) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
            return [item["citingPaper"] for item in data.get("data", [])]

    async def get_references(self, paper_id: str, limit: int = 50) -> list[dict]:
        """Get papers referenced by the given paper."""
        await self._rl.wait()
        url = f"{self._base_url}/paper/{paper_id}/references"
        params = {"fields": _S2_REFERENCE_FIELDS, "limit": min(limit, 1000)}
        async with httpx.AsyncClient(timeout=30, headers=self._headers) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
            return [item["citedPaper"] for item in data.get("data", [])]

    @staticmethod
    def s2_paper_to_paper(data: dict) -> Paper:
        """Convert S2 API response to Paper dataclass."""
        ext = data.get("externalIds") or {}
        doi = ext.get("DOI") or (f"arxiv:{ext['ArXiv']}" if ext.get("ArXiv") else None)
        arxiv_id = ext.get("ArXiv")

        if not doi:
            s2_id = data.get("paperId", "")
            doi = f"s2:{s2_id}"

        authors = [a.get("name", "") for a in data.get("authors", [])]
        year = data.get("year") or ""
        pub_date = data.get("publicationDate") or (f"{year}-01-01" if year else "")

        oa = data.get("openAccessPdf") or {}
        pdf_url = oa.get("url")

        return Paper(
            doi=doi,
            arxiv_id=arxiv_id,
            title=data.get("title", ""),
            abstract=data.get("abstract", "") or "",
            authors=authors,
            published_date=pub_date,
            updated_date=None,
            pdf_url=pdf_url,
            categories=data.get("fieldsOfStudy") or [],
            citation_count=data.get("citationCount"),
            reference_count=data.get("referenceCount"),
            s2_paper_id=data.get("paperId"),
        )


# ---------------------------------------------------------------------------
# Embedding generator
# ---------------------------------------------------------------------------

_model_cache: dict[str, Any] = {}


def get_embeddings(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    """Generate embeddings synchronously using SentenceTransformers."""
    if not _ST_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed")

    cfg = _load_config()
    name = model_name or cfg["embeddings"]["model"]

    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)

    model = _model_cache[name]
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [v.tolist() for v in vectors]
