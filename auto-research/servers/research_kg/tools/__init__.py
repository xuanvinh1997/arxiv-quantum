"""KG CRUD, search, and graph-traversal operations backed by Neo4j."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .neo4j_client import Neo4jClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _claim_id(paper_doi: str, text: str) -> str:
    digest = hashlib.sha256(f"{paper_doi}:{text}".encode()).hexdigest()[:16]
    return f"claim:{digest}"


def _result_id(paper_doi: str, metric: str, value: str) -> str:
    digest = hashlib.sha256(f"{paper_doi}:{metric}:{value}".encode()).hexdigest()[:16]
    return f"result:{digest}"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

async def init_schema(client: Neo4jClient) -> dict:
    """Apply constraints, indices, fulltext indices, and vector index."""
    from .schema import CONSTRAINTS, INDICES, FULLTEXT_INDICES, VECTOR_INDEX

    all_queries = CONSTRAINTS + INDICES + FULLTEXT_INDICES + [VECTOR_INDEX]
    results = []
    for cypher in all_queries:
        try:
            await client.run_write(cypher.strip())
            results.append({"query": cypher[:60], "status": "ok"})
        except Exception as e:
            results.append({"query": cypher[:60], "status": "error", "error": str(e)})
    return {"applied": len(results), "details": results}


# ---------------------------------------------------------------------------
# Paper CRUD
# ---------------------------------------------------------------------------

_UPSERT_PAPER_CYPHER = """
MERGE (p:Paper {doi: $doi})
SET p.arxiv_id       = $arxiv_id,
    p.title          = $title,
    p.abstract       = $abstract,
    p.published_date = $published_date,
    p.updated_date   = $updated_date,
    p.pdf_url        = $pdf_url,
    p.categories     = $categories,
    p.citation_count = $citation_count,
    p.s2_paper_id    = $s2_paper_id
WITH p
UNWIND $authors AS author_name
  MERGE (a:Author {name: author_name})
  MERGE (a)-[:AUTHORED]->(p)
WITH p
UNWIND $topics AS topic_name
  MERGE (t:Topic {name: topic_name})
  MERGE (p)-[:COVERS]->(t)
RETURN p.doi AS doi
"""

_UPSERT_EMBEDDING_CYPHER = """
MATCH (p:Paper {doi: $doi})
CALL db.create.setNodeVectorProperty(p, 'embedding', $embedding)
RETURN p.doi AS doi
"""


async def upsert_paper(client: Neo4jClient, paper: dict) -> dict:
    """Insert or update a single Paper node with authors and topic links."""
    params = {
        "doi": paper["doi"],
        "arxiv_id": paper.get("arxiv_id"),
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "published_date": paper.get("published_date", ""),
        "updated_date": paper.get("updated_date"),
        "pdf_url": paper.get("pdf_url"),
        "categories": paper.get("categories", []),
        "citation_count": paper.get("citation_count"),
        "s2_paper_id": paper.get("s2_paper_id"),
        "authors": paper.get("authors", []),
        "topics": paper.get("topics", paper.get("categories", [])),
    }
    result = await client.run_write(_UPSERT_PAPER_CYPHER, params)

    if paper.get("embedding"):
        await client.run_write(
            _UPSERT_EMBEDDING_CYPHER,
            {"doi": paper["doi"], "embedding": paper["embedding"]},
        )
        result["embedding_set"] = True

    return result


async def upsert_papers(client: Neo4jClient, papers: list[dict]) -> dict:
    """Upsert multiple papers; returns aggregate counters."""
    totals: dict[str, int] = {"nodes_created": 0, "relationships_created": 0, "properties_set": 0}
    for paper in papers:
        r = await upsert_paper(client, paper)
        for k in totals:
            totals[k] += r.get(k, 0)
    totals["papers_processed"] = len(papers)
    return totals


# ---------------------------------------------------------------------------
# Citation edges
# ---------------------------------------------------------------------------

_UPSERT_CITES_CYPHER = """
MATCH (from:Paper {doi: $from_doi})
MATCH (to:Paper   {doi: $to_doi})
MERGE (from)-[r:CITES]->(to)
SET r.context = $context
RETURN type(r) AS rel
"""


async def add_citation(
    client: Neo4jClient,
    from_doi: str,
    to_doi: str,
    context: str = "",
) -> dict:
    """Create a CITES relationship between two Paper nodes.

    Both papers must already exist in the KG.
    """
    result = await client.run_query(
        _UPSERT_CITES_CYPHER,
        {"from_doi": from_doi, "to_doi": to_doi, "context": context},
    )
    return {"success": bool(result), "from": from_doi, "to": to_doi}


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

_UPSERT_CLAIM_CYPHER = """
MERGE (c:Claim {id: $id})
SET c.text            = $text,
    c.claim_type      = $claim_type,
    c.subject         = $subject,
    c.predicate       = $predicate,
    c.object          = $object,
    c.confidence      = $confidence,
    c.source_paper_doi = $paper_doi
WITH c
MATCH (p:Paper {doi: $paper_doi})
MERGE (p)-[:ASSERTS {section: $section}]->(c)
WITH c
UNWIND $topics AS topic_name
  MERGE (t:Topic {name: topic_name})
  MERGE (c)-[:COVERS]->(t)
RETURN c.id AS id
"""


async def add_claim(client: Neo4jClient, claim: dict) -> dict:
    """Insert or update a Claim node and link it to its source Paper.

    Required keys: paper_doi, text
    Optional keys: claim_type, subject, predicate, object, confidence, section, topics
    """
    paper_doi = claim["paper_doi"]
    text = claim["text"]
    claim_id = claim.get("id") or _claim_id(paper_doi, text)

    params = {
        "id": claim_id,
        "text": text,
        "claim_type": claim.get("claim_type", "general"),
        "subject": claim.get("subject", ""),
        "predicate": claim.get("predicate", ""),
        "object": claim.get("object", ""),
        "confidence": claim.get("confidence", 1.0),
        "paper_doi": paper_doi,
        "section": claim.get("section", ""),
        "topics": claim.get("topics", []),
    }
    await client.run_write(_UPSERT_CLAIM_CYPHER, params)
    return {"id": claim_id, "paper_doi": paper_doi}


async def add_claim_relationship(
    client: Neo4jClient,
    from_claim_id: str,
    to_claim_id: str,
    rel_type: str,
    props: dict | None = None,
) -> dict:
    """Add SUPPORTS or CONTRADICTS between two Claim nodes.

    rel_type must be "SUPPORTS" or "CONTRADICTS".
    """
    if rel_type not in ("SUPPORTS", "CONTRADICTS"):
        raise ValueError("rel_type must be SUPPORTS or CONTRADICTS")

    props = props or {}
    set_clause = " ".join(f"r.{k} = ${k}" for k in props) if props else "r._t = 1"
    cypher = f"""
    MATCH (a:Claim {{id: $from_id}})
    MATCH (b:Claim {{id: $to_id}})
    MERGE (a)-[r:{rel_type}]->(b)
    SET {set_clause}
    RETURN type(r) AS rel
    """
    params = {"from_id": from_claim_id, "to_id": to_claim_id, **props}
    result = await client.run_query(cypher, params)
    return {"success": bool(result), "rel_type": rel_type}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

_UPSERT_RESULT_CYPHER = """
MERGE (r:Result {id: $id})
SET r.metric_name   = $metric_name,
    r.metric_value  = $metric_value,
    r.metric_unit   = $metric_unit,
    r.context       = $context,
    r.source_paper_doi = $paper_doi
WITH r
MATCH (p:Paper {doi: $paper_doi})
MERGE (p)-[:REPORTS]->(r)
WITH r
OPTIONAL MATCH (m:Method {name: $method_name})
FOREACH (_ IN CASE WHEN m IS NOT NULL THEN [1] ELSE [] END |
  MERGE (r)-[:MEASURES]->(m)
)
RETURN r.id AS id
"""


async def add_result(client: Neo4jClient, result_data: dict) -> dict:
    """Record a numerical result reported in a paper.

    Required keys: paper_doi, metric_name, metric_value
    Optional keys: metric_unit, context, method_name
    """
    paper_doi = result_data["paper_doi"]
    metric = result_data["metric_name"]
    value = str(result_data["metric_value"])
    result_id = result_data.get("id") or _result_id(paper_doi, metric, value)

    params = {
        "id": result_id,
        "metric_name": metric,
        "metric_value": value,
        "metric_unit": result_data.get("metric_unit", ""),
        "context": result_data.get("context", ""),
        "paper_doi": paper_doi,
        "method_name": result_data.get("method_name", ""),
    }
    await client.run_write(_UPSERT_RESULT_CYPHER, params)
    return {"id": result_id}


# ---------------------------------------------------------------------------
# Method nodes
# ---------------------------------------------------------------------------

_UPSERT_METHOD_CYPHER = """
MERGE (m:Method {name: $name})
SET m.category    = $category,
    m.description = $description
RETURN m.name AS name
"""


async def upsert_method(client: Neo4jClient, method: dict) -> dict:
    """Insert or update a Method node."""
    params = {
        "name": method["name"],
        "category": method.get("category", ""),
        "description": method.get("description", ""),
    }
    await client.run_write(_UPSERT_METHOD_CYPHER, params)
    return {"name": method["name"]}


async def link_paper_method(
    client: Neo4jClient,
    paper_doi: str,
    method_name: str,
    variant: str = "",
    configuration: str = "",
) -> dict:
    """Create USES_METHOD edge between a Paper and a Method."""
    cypher = """
    MATCH (p:Paper  {doi: $doi})
    MATCH (m:Method {name: $name})
    MERGE (p)-[r:USES_METHOD]->(m)
    SET r.variant = $variant, r.configuration = $configuration
    RETURN type(r) AS rel
    """
    result = await client.run_query(
        cypher,
        {"doi": paper_doi, "name": method_name, "variant": variant, "configuration": configuration},
    )
    return {"success": bool(result)}


# ---------------------------------------------------------------------------
# Read / search
# ---------------------------------------------------------------------------

async def get_paper(client: Neo4jClient, identifier: str) -> dict | None:
    """Fetch a Paper node by DOI or arXiv ID."""
    cypher = """
    MATCH (p:Paper)
    WHERE p.doi = $id OR p.arxiv_id = $id
    OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
    OPTIONAL MATCH (p)-[:COVERS]->(t:Topic)
    RETURN p {.*} AS paper,
           collect(DISTINCT a.name) AS authors,
           collect(DISTINCT t.name) AS topics
    LIMIT 1
    """
    rows = await client.run_query(cypher, {"id": identifier})
    if not rows:
        return None
    row = rows[0]
    result = row["paper"]
    result["authors"] = row["authors"]
    result["topics"] = row["topics"]
    return result


async def search_papers_fulltext(
    client: Neo4jClient,
    query: str,
    limit: int = 10,
) -> list[dict]:
    """Full-text search over paper titles and abstracts."""
    cypher = """
    CALL db.index.fulltext.queryNodes('paper_text_idx', $query)
    YIELD node AS p, score
    RETURN p {.*, score: score} AS paper
    ORDER BY score DESC
    LIMIT $limit
    """
    rows = await client.run_query(cypher, {"query": query, "limit": limit})
    return [r["paper"] for r in rows]


async def search_papers_vector(
    client: Neo4jClient,
    embedding: list[float],
    limit: int = 10,
) -> list[dict]:
    """Vector similarity search over paper embeddings."""
    cypher = """
    CALL db.index.vector.queryNodes('paper_embedding_idx', $limit, $embedding)
    YIELD node AS p, score
    RETURN p {.*, score: score} AS paper
    ORDER BY score DESC
    """
    rows = await client.run_query(cypher, {"embedding": embedding, "limit": limit})
    return [r["paper"] for r in rows]


async def get_topic_papers(
    client: Neo4jClient,
    topic_name: str,
    limit: int = 20,
) -> list[dict]:
    """Return papers covering a given topic (exact or fuzzy name match)."""
    cypher = """
    MATCH (t:Topic)
    WHERE t.name =~ ('(?i).*' + $topic + '.*')
    MATCH (p:Paper)-[:COVERS]->(t)
    RETURN p {.*} AS paper, t.name AS topic
    ORDER BY p.published_date DESC
    LIMIT $limit
    """
    rows = await client.run_query(cypher, {"topic": topic_name, "limit": limit})
    return [r["paper"] for r in rows]


async def get_citation_network(
    client: Neo4jClient,
    doi: str,
    depth: int = 2,
    limit: int = 50,
) -> dict:
    """Return the citation subgraph around a paper up to *depth* hops.

    Returns {"nodes": [...], "edges": [...]} where each node is a paper dict
    and each edge is {"from": doi, "to": doi}.
    """
    depth = min(depth, 4)
    cypher = """
    MATCH path = (p:Paper {doi: $doi})-[:CITES*1..$depth]-(q:Paper)
    WITH p, q, relationships(path) AS rels
    RETURN DISTINCT q {.*} AS paper,
           [r IN rels | {from: startNode(r).doi, to: endNode(r).doi}] AS edges
    LIMIT $limit
    """
    rows = await client.run_query(cypher, {"doi": doi, "depth": depth, "limit": limit})

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for row in rows:
        paper = row["paper"]
        nodes[paper["doi"]] = paper
        for edge in row["edges"]:
            if edge not in edges:
                edges.append(edge)

    # Include seed paper
    seed = await get_paper(client, doi)
    if seed:
        nodes[doi] = seed

    return {"nodes": list(nodes.values()), "edges": edges}


async def get_paper_claims(client: Neo4jClient, doi: str) -> list[dict]:
    """Return all claims asserted by a paper."""
    cypher = """
    MATCH (p:Paper {doi: $doi})-[:ASSERTS]->(c:Claim)
    OPTIONAL MATCH (c)-[:COVERS]->(t:Topic)
    RETURN c {.*} AS claim, collect(t.name) AS topics
    """
    rows = await client.run_query(cypher, {"doi": doi})
    out = []
    for row in rows:
        c = row["claim"]
        c["topics"] = row["topics"]
        out.append(c)
    return out


async def get_contradictions(
    client: Neo4jClient,
    min_confidence: float = 0.5,
) -> list[dict]:
    """Return pairs of contradicting claims with their source papers."""
    cypher = """
    MATCH (a:Claim)-[r:CONTRADICTS]->(b:Claim)
    WHERE r.confidence >= $min_conf OR r.confidence IS NULL
    MATCH (pa:Paper)-[:ASSERTS]->(a)
    MATCH (pb:Paper)-[:ASSERTS]->(b)
    RETURN a {.*} AS claim_a, b {.*} AS claim_b,
           pa.doi AS paper_a_doi, pa.title AS paper_a_title,
           pb.doi AS paper_b_doi, pb.title AS paper_b_title,
           r.explanation AS explanation, r.confidence AS confidence
    LIMIT 100
    """
    rows = await client.run_query(cypher, {"min_conf": min_confidence})
    return rows


async def get_kg_stats(client: Neo4jClient) -> dict:
    """Return counts of all node and relationship types in the KG."""
    node_cypher = "CALL apoc.meta.stats() YIELD labels RETURN labels"
    rel_cypher = "CALL apoc.meta.stats() YIELD relTypesCount RETURN relTypesCount"

    fallback_cypher = """
    MATCH (n) RETURN labels(n) AS lbl, count(*) AS cnt
    """
    try:
        node_rows = await client.run_query(node_cypher)
        rel_rows = await client.run_query(rel_cypher)
        return {
            "labels": node_rows[0]["labels"] if node_rows else {},
            "rel_types": rel_rows[0]["relTypesCount"] if rel_rows else {},
        }
    except Exception:
        rows = await client.run_query(fallback_cypher)
        counts: dict[str, int] = {}
        for row in rows:
            for lbl in row["lbl"]:
                counts[lbl] = counts.get(lbl, 0) + row["cnt"]
        return {"labels": counts, "rel_types": {}}


async def run_read_cypher(
    client: Neo4jClient,
    cypher: str,
    params: dict | None = None,
) -> list[dict]:
    """Execute an arbitrary read-only Cypher query and return results.

    Only MATCH / RETURN queries are allowed; write clauses will be rejected.
    """
    forbidden = {"CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DETACH"}
    upper = cypher.upper()
    for kw in forbidden:
        if kw in upper:
            raise ValueError(f"Write clause '{kw}' not allowed in read queries")
    return await client.run_query(cypher, params or {})
