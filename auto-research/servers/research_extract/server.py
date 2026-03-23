"""MCP server: research_extract — LLM-powered extraction of claims, methods, results."""
from __future__ import annotations

import asyncio

from fastmcp import FastMCP

from .tools import (
    extract_claims,
    extract_methods,
    extract_results,
    extract_full,
    detect_contradictions_llm,
)

# Import KG tools lazily to avoid requiring Neo4j when only doing extraction
_kg_client = None


def _get_kg():
    global _kg_client
    if _kg_client is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from servers.research_kg.neo4j_client import Neo4jClient
        _kg_client = Neo4jClient.from_config()
    return _kg_client


mcp = FastMCP("research_extract")


# ---------------------------------------------------------------------------
# Extraction-only tools (no KG required)
# ---------------------------------------------------------------------------

@mcp.tool()
def extract_paper_claims(
    title: str,
    abstract: str,
    doi: str = "",
    model: str | None = None,
) -> list[dict]:
    """Extract scientific claims (SPO triples) from a paper using Claude.

    Returns a list of claim dicts. Each claim has:
      text, claim_type, subject, predicate, object, confidence, section, paper_doi

    claim_type: result | method | hypothesis | comparison | limitation | general

    Pass the output directly to kg_add_claim to store in the KG.
    Tip: use model="claude-haiku-4-5" for faster, cheaper bulk extraction.
    """
    return extract_claims(title=title, abstract=abstract, doi=doi, model=model)


@mcp.tool()
def extract_paper_methods(
    title: str,
    abstract: str,
    doi: str = "",
    model: str | None = None,
) -> list[dict]:
    """Extract algorithms and methods mentioned in a paper using Claude.

    Returns a list of method dicts. Each has:
      name, category, variant, description, paper_doi

    category: ansatz | optimizer | error_correction | noise_mitigation |
              dataset | framework | classical_baseline | metric | other

    Pass name+category+description to kg_upsert_method, then
    kg_link_paper_method(doi, name) to wire the USES_METHOD edge.
    """
    return extract_methods(title=title, abstract=abstract, doi=doi, model=model)


@mcp.tool()
def extract_paper_results(
    title: str,
    abstract: str,
    doi: str = "",
    model: str | None = None,
) -> list[dict]:
    """Extract quantitative results reported in a paper using Claude.

    Returns a list of result dicts. Each has:
      metric_name, metric_value, metric_unit, context, method_name, paper_doi

    Pass directly to kg_add_result to store in the KG.
    Returns empty list if no explicit numerical results are found.
    """
    return extract_results(title=title, abstract=abstract, doi=doi, model=model)


@mcp.tool()
def extract_paper_full(
    title: str,
    abstract: str,
    doi: str = "",
    model: str | None = None,
) -> dict:
    """Extract claims, methods, AND results in a single Claude call.

    More token-efficient than three separate calls (one API round-trip).
    Uses adaptive thinking for higher extraction quality.

    Returns {"claims": [...], "methods": [...], "results": [...]}

    Each sub-list is ready for the corresponding KG write tool.
    """
    return extract_full(title=title, abstract=abstract, doi=doi, model=model)


# ---------------------------------------------------------------------------
# Integrated tools (extraction + KG storage in one shot)
# ---------------------------------------------------------------------------

@mcp.tool()
async def process_paper(
    doi: str,
    model: str | None = None,
    store_in_kg: bool = True,
) -> dict:
    """Fetch a paper from the KG, extract its claims/methods/results, and store them back.

    The paper must already exist in the KG (use kg_upsert_paper first).

    Returns a summary of what was extracted and stored.
    Set store_in_kg=False for a dry run that returns extracted data without writing.
    """
    from servers.research_kg.tools import (
        get_paper,
        add_claim,
        add_result,
        upsert_method,
        link_paper_method,
    )

    kg = _get_kg()
    paper = await get_paper(kg, doi)
    if paper is None:
        return {"error": f"Paper not found in KG: {doi}"}

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    if not abstract:
        return {"error": f"Paper {doi} has no abstract in KG"}

    # Extract all in one call
    loop = asyncio.get_event_loop()
    extracted = await loop.run_in_executor(
        None, lambda: extract_full(title=title, abstract=abstract, doi=doi, model=model)
    )

    summary = {
        "doi": doi,
        "title": title,
        "claims_found": len(extracted["claims"]),
        "methods_found": len(extracted["methods"]),
        "results_found": len(extracted["results"]),
        "stored": store_in_kg,
    }

    if store_in_kg:
        claim_ids = []
        for claim in extracted["claims"]:
            result = await add_claim(kg, claim)
            claim_ids.append(result["id"])
        summary["claim_ids"] = claim_ids

        for method in extracted["methods"]:
            await upsert_method(kg, {"name": method["name"], "category": method["category"], "description": method["description"]})
            await link_paper_method(kg, doi, method["name"], variant=method.get("variant", ""))
        summary["methods_stored"] = [m["name"] for m in extracted["methods"]]

        for res in extracted["results"]:
            await add_result(kg, res)
        summary["results_stored"] = len(extracted["results"])
    else:
        summary["data"] = extracted

    return summary


@mcp.tool()
async def batch_process_papers(
    dois: list[str],
    model: str | None = None,
    concurrency: int = 3,
) -> list[dict]:
    """Process multiple papers: extract claims/methods/results and store in KG.

    Papers must already exist in the KG.
    *concurrency* controls how many papers are processed in parallel (default 3).

    Tip: use model="claude-haiku-4-5" for much cheaper bulk processing.
    Returns a list of per-paper summaries.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _process_one(doi: str) -> dict:
        async with sem:
            return await process_paper(doi=doi, model=model, store_in_kg=True)

    tasks = [_process_one(doi) for doi in dois]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    summaries = []
    for doi, result in zip(dois, results):
        if isinstance(result, Exception):
            summaries.append({"doi": doi, "error": str(result)})
        else:
            summaries.append(result)
    return summaries


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

@mcp.tool()
async def find_contradictions_for_paper(
    doi: str,
    min_kg_confidence: float = 0.0,
    llm_verify: bool = True,
    model: str | None = None,
) -> list[dict]:
    """Find claims from other papers that may contradict the given paper's claims.

    Step 1: Query KG for existing CONTRADICTS edges (fast, no LLM).
    Step 2: If llm_verify=True, use Claude to re-evaluate edge cases.

    Returns a list of contradiction records with explanation and confidence.
    """
    from servers.research_kg.tools import get_paper_claims, get_contradictions

    kg = _get_kg()
    contradictions = await get_contradictions(kg, min_confidence=min_kg_confidence)

    # Filter to those involving our paper
    paper_claims = await get_paper_claims(kg, doi)
    paper_claim_ids = {c.get("id") for c in paper_claims if c.get("id")}

    relevant = [
        c for c in contradictions
        if c.get("paper_a_doi") == doi or c.get("paper_b_doi") == doi
    ]

    if not llm_verify or not relevant:
        return relevant

    # Re-verify with LLM (run in thread pool to avoid blocking)
    loop = asyncio.get_event_loop()
    verified = []
    for contra in relevant:
        claim_a = {"text": contra.get("claim_a", {}).get("text", ""), "source_paper_doi": contra.get("paper_a_doi", "")}
        claim_b = {"text": contra.get("claim_b", {}).get("text", ""), "source_paper_doi": contra.get("paper_b_doi", "")}

        judgement = await loop.run_in_executor(
            None, lambda a=claim_a, b=claim_b: detect_contradictions_llm(a, b, model=model)
        )

        contra["llm_verified"] = judgement["contradicts"]
        contra["llm_confidence"] = judgement["confidence"]
        contra["llm_explanation"] = judgement["explanation"]
        verified.append(contra)

    return verified


if __name__ == "__main__":
    mcp.run()
