"""MCP server: research_fetch — arXiv & Semantic Scholar paper fetching."""
from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from .tools import (
    ArxivClient,
    SemanticScholarClient,
    get_embeddings,
)

mcp = FastMCP("research_fetch")

_arxiv = ArxivClient.from_config()
_s2 = SemanticScholarClient.from_config()


# ---------------------------------------------------------------------------
# arXiv tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_arxiv(
    query: str,
    max_results: int = 20,
    categories: list[str] | None = None,
    sort_by: str = "relevance",
) -> list[dict]:
    """Search arXiv for papers matching *query*.

    Args:
        query: Free-text or arXiv advanced search query.
        max_results: Maximum papers to return (default 20, max 200).
        categories: Optional list of arXiv category filters e.g. ["quant-ph", "cs.LG"].
        sort_by: "relevance" (default) or "lastUpdatedDate" or "submittedDate".
    """
    papers = await _arxiv.search(
        query, max_results=min(max_results, 200), categories=categories, sort_by=sort_by
    )
    return [p.to_dict() for p in papers]


@mcp.tool()
async def get_arxiv_paper(arxiv_id: str) -> dict | None:
    """Fetch metadata for a single arXiv paper by its ID (e.g. "2401.12345").

    Returns None if the paper is not found.
    """
    paper = await _arxiv.get_paper(arxiv_id)
    return paper.to_dict() if paper else None


# ---------------------------------------------------------------------------
# Semantic Scholar tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_semantic_scholar(
    query: str,
    limit: int = 20,
) -> list[dict]:
    """Search Semantic Scholar for papers matching *query*.

    Returns richer metadata than arXiv (citation counts, references, etc.).
    """
    results = await _s2.search(query, limit=min(limit, 100))
    papers = [SemanticScholarClient.s2_paper_to_paper(r).to_dict() for r in results]
    return papers


@mcp.tool()
async def get_s2_paper(paper_id: str) -> dict | None:
    """Get full metadata for one paper from Semantic Scholar.

    *paper_id* can be:
    - Semantic Scholar ID  (e.g. "649def34f8be52c8b66281af98ae884c09aef38b")
    - DOI                  (e.g. "10.18653/v1/2020.acl-main.463")
    - ArXiv ID with prefix (e.g. "ArXiv:2401.12345")
    - URL                  (e.g. "https://arxiv.org/abs/2401.12345")
    """
    data = await _s2.get_paper(paper_id)
    if data is None:
        return None
    return SemanticScholarClient.s2_paper_to_paper(data).to_dict()


@mcp.tool()
async def get_paper_citations(paper_id: str, limit: int = 50) -> list[dict]:
    """Return up to *limit* papers that cite the given paper.

    *paper_id* accepts the same formats as get_s2_paper.
    """
    items = await _s2.get_citations(paper_id, limit=limit)
    return [SemanticScholarClient.s2_paper_to_paper(i).to_dict() for i in items if i.get("title")]


@mcp.tool()
async def get_paper_references(paper_id: str, limit: int = 50) -> list[dict]:
    """Return up to *limit* papers that the given paper references.

    *paper_id* accepts the same formats as get_s2_paper.
    """
    items = await _s2.get_references(paper_id, limit=limit)
    return [SemanticScholarClient.s2_paper_to_paper(i).to_dict() for i in items if i.get("title")]


# ---------------------------------------------------------------------------
# Embedding tool
# ---------------------------------------------------------------------------

@mcp.tool()
def embed_texts(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    """Generate sentence embeddings for a list of texts.

    Uses the model configured in config.toml (default: all-MiniLM-L6-v2, 384-dim).
    Returns a list of float vectors, one per input text.
    """
    return get_embeddings(texts, model_name=model_name)


if __name__ == "__main__":
    mcp.run()
