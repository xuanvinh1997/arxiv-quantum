# auto-research — Claude context

## What this is
Two MCP servers that together form an automated research assistant for quantum ML papers:
- **research_fetch**: fetches papers from arXiv and Semantic Scholar
- **research_kg**: stores and queries a Neo4j knowledge graph of papers, claims, methods, and results

## Running the servers
```bash
mcp dev servers/research_fetch/server.py
mcp dev servers/research_kg/server.py
```

Neo4j must be running first (`docker compose -f docker-compose.neo4j.yml up -d`).

## Key file locations
- `config/config.toml` — all service config (Neo4j creds, API keys, embedding model)
- `config/topics.yaml` — research topics to track (used by seed_kg.py)
- `servers/research_kg/schema.py` — Neo4j node/edge schema documentation
- `scripts/seed_kg.py` — populate KG from arXiv; set `SEED_SKIP_TOPICS=1` to skip topic searches

## Python environment
`/mnt/hdd/miniconda3/envs/research/bin/python` (see setup.sh)

## Repo papers seeded into the KG
- `2601.07223` — quantum classifier implementation (arxiv_2601_07223/)
- `2602.17615` — Shadow Enhanced Greedy Quantum Eigensolver (arxiv_2602_17615/)
