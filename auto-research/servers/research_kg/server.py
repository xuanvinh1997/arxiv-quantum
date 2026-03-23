"""MCP server: research_kg — knowledge-graph CRUD and traversal."""
from __future__ import annotations

from fastmcp import FastMCP

from .neo4j_client import Neo4jClient
from .tools import (
    init_schema,
    upsert_paper,
    upsert_papers,
    add_citation,
    add_claim,
    add_claim_relationship,
    add_result,
    upsert_method,
    link_paper_method,
    get_paper,
    search_papers_fulltext,
    search_papers_vector,
    get_topic_papers,
    get_citation_network,
    get_paper_claims,
    get_contradictions,
    get_kg_stats,
    run_read_cypher,
)

mcp = FastMCP("research_kg")

# Shared client — created lazily on first request
_client: Neo4jClient | None = None


def _get_client() -> Neo4jClient:
    global _client
    if _client is None:
        _client = Neo4jClient.from_config()
    return _client


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@mcp.tool()
async def kg_init_schema() -> dict:
    """Apply Neo4j constraints, indices, fulltext indices, and vector index.

    Safe to call multiple times (uses IF NOT EXISTS). Run once after Neo4j starts.
    """
    return await init_schema(_get_client())


# ---------------------------------------------------------------------------
# Paper write tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def kg_upsert_paper(paper: dict) -> dict:
    """Insert or update a single Paper node in the knowledge graph.

    Required keys: doi, title
    Typical keys: arxiv_id, abstract, authors (list), published_date,
                  categories (list), pdf_url, embedding (list[float]),
                  citation_count, s2_paper_id, topics (list)
    """
    return await upsert_paper(_get_client(), paper)


@mcp.tool()
async def kg_upsert_papers(papers: list[dict]) -> dict:
    """Bulk upsert a list of Paper nodes. Returns aggregate counters."""
    return await upsert_papers(_get_client(), papers)


@mcp.tool()
async def kg_add_citation(
    from_doi: str,
    to_doi: str,
    context: str = "",
) -> dict:
    """Record that the paper *from_doi* cites the paper *to_doi*.

    Both papers must already exist in the KG (use kg_upsert_paper first).
    *context* is an optional excerpt describing the citation.
    """
    return await add_citation(_get_client(), from_doi, to_doi, context)


# ---------------------------------------------------------------------------
# Claim write tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def kg_add_claim(claim: dict) -> dict:
    """Add a scientific claim extracted from a paper.

    Required keys: paper_doi, text
    Optional keys: claim_type (e.g. "result"|"method"|"hypothesis"),
                   subject, predicate, object (SPO triple),
                   confidence (0-1), section, topics (list)
    """
    return await add_claim(_get_client(), claim)


@mcp.tool()
async def kg_add_claim_relationship(
    from_claim_id: str,
    to_claim_id: str,
    rel_type: str,
    props: dict | None = None,
) -> dict:
    """Connect two claims with SUPPORTS or CONTRADICTS.

    *props* can include: strength (float), explanation (str), confidence (float).
    """
    return await add_claim_relationship(_get_client(), from_claim_id, to_claim_id, rel_type, props)


# ---------------------------------------------------------------------------
# Result & Method tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def kg_add_result(result_data: dict) -> dict:
    """Record a numerical result reported in a paper.

    Required keys: paper_doi, metric_name, metric_value
    Optional keys: metric_unit, context, method_name
    """
    return await add_result(_get_client(), result_data)


@mcp.tool()
async def kg_upsert_method(method: dict) -> dict:
    """Insert or update a Method node.

    Required keys: name
    Optional keys: category (e.g. "ansatz"|"optimizer"|"error_correction"),
                   description
    """
    return await upsert_method(_get_client(), method)


@mcp.tool()
async def kg_link_paper_method(
    paper_doi: str,
    method_name: str,
    variant: str = "",
    configuration: str = "",
) -> dict:
    """Create a USES_METHOD edge between a Paper and a Method node."""
    return await link_paper_method(_get_client(), paper_doi, method_name, variant, configuration)


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def kg_get_paper(identifier: str) -> dict | None:
    """Fetch a Paper node by its DOI or arXiv ID.

    Returns the paper with its authors and topics, or None if not found.
    """
    return await get_paper(_get_client(), identifier)


@mcp.tool()
async def kg_search_fulltext(query: str, limit: int = 10) -> list[dict]:
    """Full-text search over paper titles and abstracts.

    Uses the Neo4j fulltext index; supports Lucene query syntax.
    """
    return await search_papers_fulltext(_get_client(), query, limit)


@mcp.tool()
async def kg_search_vector(embedding: list[float], limit: int = 10) -> list[dict]:
    """Find papers by cosine similarity to *embedding* (384-dim vector).

    Use embed_texts from the research_fetch server to generate embeddings.
    """
    return await search_papers_vector(_get_client(), embedding, limit)


@mcp.tool()
async def kg_get_topic_papers(topic_name: str, limit: int = 20) -> list[dict]:
    """Return papers tagged with a topic (case-insensitive substring match)."""
    return await get_topic_papers(_get_client(), topic_name, limit)


@mcp.tool()
async def kg_get_citation_network(doi: str, depth: int = 2, limit: int = 50) -> dict:
    """Return the citation subgraph around *doi* up to *depth* hops.

    Returns {"nodes": [paper, ...], "edges": [{"from": doi, "to": doi}, ...]}.
    """
    return await get_citation_network(_get_client(), doi, depth, limit)


@mcp.tool()
async def kg_get_paper_claims(doi: str) -> list[dict]:
    """Return all claims asserted by the paper identified by *doi*."""
    return await get_paper_claims(_get_client(), doi)


@mcp.tool()
async def kg_get_contradictions(min_confidence: float = 0.0) -> list[dict]:
    """Return pairs of contradicting claims across the knowledge graph.

    Filter by *min_confidence* (0–1) on the CONTRADICTS relationship.
    """
    return await get_contradictions(_get_client(), min_confidence)


@mcp.tool()
async def kg_stats() -> dict:
    """Return node and relationship type counts from the knowledge graph."""
    return await get_kg_stats(_get_client())


@mcp.tool()
async def kg_cypher(query: str, params: dict | None = None) -> list[dict]:
    """Execute a read-only Cypher query and return results as a list of dicts.

    Only MATCH / RETURN queries are accepted; write clauses are rejected.
    """
    return await run_read_cypher(_get_client(), query, params)


if __name__ == "__main__":
    mcp.run()
