#!/usr/bin/env python3
"""Seed the KG with papers already in the repository and topic-based searches.

This script:
1. Fetches metadata for known arXiv IDs present in this repo.
2. Searches arXiv for each configured topic and ingests results.
3. Generates embeddings (optional, requires sentence-transformers).
4. Upserts everything into Neo4j.
"""
import asyncio
import os
import sys
from pathlib import Path

# Allow running from project root or scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from servers.research_fetch.tools import ArxivClient, get_embeddings
from servers.research_kg.neo4j_client import Neo4jClient
from servers.research_kg.tools import upsert_papers

# ---------------------------------------------------------------------------
# Papers known to exist in this repository
# ---------------------------------------------------------------------------

REPO_ARXIV_IDS: list[str] = [
    "2601.07223",
    "2602.17615",
]

# ---------------------------------------------------------------------------
# How many papers per topic to fetch on first seed
# ---------------------------------------------------------------------------

PAPERS_PER_TOPIC = 10

# ---------------------------------------------------------------------------

_TOPICS_PATH = Path(__file__).parent.parent / "config" / "topics.yaml"


def _load_topics() -> list[dict]:
    with open(_TOPICS_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("topics", [])


def _embed(papers: list[dict]) -> list[dict]:
    texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers]
    try:
        vectors = get_embeddings(texts)
        for paper, vec in zip(papers, vectors):
            paper["embedding"] = vec
    except Exception as exc:
        print(f"  [warn] Embeddings skipped: {exc}")
    return papers


async def seed_repo_papers(arxiv: ArxivClient, client: Neo4jClient) -> int:
    """Fetch and ingest the papers that live in this repo."""
    print(f"\n=== Seeding {len(REPO_ARXIV_IDS)} repo papers ===")
    papers = []
    for arxiv_id in REPO_ARXIV_IDS:
        print(f"  Fetching arXiv:{arxiv_id} …")
        paper = await arxiv.get_paper(arxiv_id)
        if paper:
            papers.append(paper.to_dict())
        else:
            print(f"  [warn] Not found: {arxiv_id}")

    if not papers:
        return 0

    papers = _embed(papers)
    result = await upsert_papers(client, papers)
    print(f"  Ingested {result['papers_processed']} repo papers")
    return result["papers_processed"]


async def seed_topic_papers(arxiv: ArxivClient, client: Neo4jClient) -> int:
    """Search arXiv for each configured topic and ingest results."""
    topics = _load_topics()
    print(f"\n=== Seeding papers for {len(topics)} topics ({PAPERS_PER_TOPIC} each) ===")

    total = 0
    for topic in topics:
        name = topic["name"]
        categories = topic.get("arxiv_categories", [])
        print(f"  Topic: {name!r} …")

        try:
            papers_raw = await arxiv.search(
                query=name,
                max_results=PAPERS_PER_TOPIC,
                categories=categories,
                sort_by="relevance",
            )
        except Exception as exc:
            print(f"  [error] arXiv search failed: {exc}")
            continue

        if not papers_raw:
            print("  (no results)")
            continue

        papers = [p.to_dict() for p in papers_raw]

        # Tag each paper with this topic
        for paper in papers:
            existing = set(paper.get("categories", []))
            existing.add(name)
            paper["topics"] = list(existing)

        papers = _embed(papers)
        result = await upsert_papers(client, papers)
        n = result["papers_processed"]
        total += n
        print(f"  → {n} papers ingested")

    return total


async def main() -> None:
    skip_topics = os.environ.get("SEED_SKIP_TOPICS", "").lower() in ("1", "true", "yes")
    skip_repo = os.environ.get("SEED_SKIP_REPO", "").lower() in ("1", "true", "yes")

    arxiv = ArxivClient.from_config()
    kg = Neo4jClient.from_config()

    count = 0

    if not skip_repo:
        count += await seed_repo_papers(arxiv, kg)

    if not skip_topics:
        count += await seed_topic_papers(arxiv, kg)

    await kg.close()

    print(f"\n=== Seed complete: {count} papers ingested total ===")


if __name__ == "__main__":
    asyncio.run(main())
