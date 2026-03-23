# auto-research

Automated research assistant for quantum ML literature, built on three MCP servers backed by Neo4j and Claude.

## Architecture

```
auto-research/
├── config/
│   ├── config.toml        # Neo4j, arXiv, S2, embeddings config
│   └── topics.yaml        # Research topics to track
├── servers/
│   ├── research_fetch/    # MCP: fetch papers from arXiv & Semantic Scholar
│   │   ├── server.py
│   │   └── tools/__init__.py
│   ├── research_kg/       # MCP: knowledge-graph CRUD & traversal
│   │   ├── server.py
│   │   ├── tools/__init__.py
│   │   ├── neo4j_client.py
│   │   └── schema.py
│   └── research_extract/  # MCP: Claude-powered extraction of claims/methods/results
│       ├── server.py
│       └── tools/__init__.py
├── scripts/
│   ├── init_neo4j_schema.py   # Apply Neo4j constraints/indices
│   └── seed_kg.py             # Populate KG with repo + topic papers
├── docker-compose.neo4j.yml
└── setup.sh
```

## Quick start

```bash
cd auto-research
./setup.sh          # installs deps, starts Neo4j, inits schema, seeds KG
```

Or step by step:

```bash
# 1. Start Neo4j
docker compose -f docker-compose.neo4j.yml up -d

# 2. Install Python deps
pip install "mcp[cli]>=1.0.0" fastmcp neo4j feedparser httpx aiofiles tomli pyyaml sentence-transformers "anthropic>=0.50.0"

# 3. Init schema
python scripts/init_neo4j_schema.py

# 4. Seed KG
python scripts/seed_kg.py           # fetch repo papers + all topic searches
SEED_SKIP_TOPICS=1 python scripts/seed_kg.py   # repo papers only

# 5. Launch MCP servers
mcp dev servers/research_fetch/server.py
mcp dev servers/research_kg/server.py
mcp dev servers/research_extract/server.py
```

Neo4j browser: http://localhost:7474 (neo4j / researchpw)

## MCP tools

### research_fetch

| Tool | Description |
|------|-------------|
| `search_arxiv` | Search arXiv with optional category filters |
| `get_arxiv_paper` | Fetch one paper by arXiv ID |
| `search_semantic_scholar` | Search S2 (citation counts included) |
| `get_s2_paper` | Get full S2 metadata by DOI / S2 ID / ArXiv prefix |
| `get_paper_citations` | Papers that cite a given paper |
| `get_paper_references` | Papers referenced by a given paper |
| `embed_texts` | Generate 384-dim embeddings (all-MiniLM-L6-v2) |

### research_kg

| Tool | Description |
|------|-------------|
| `kg_init_schema` | Apply constraints, indices, vector index |
| `kg_upsert_paper` | Insert/update a Paper node |
| `kg_upsert_papers` | Bulk upsert papers |
| `kg_add_citation` | Add CITES edge between two papers |
| `kg_add_claim` | Record a scientific claim from a paper |
| `kg_add_claim_relationship` | Connect claims with SUPPORTS/CONTRADICTS |
| `kg_add_result` | Record a numerical result from a paper |
| `kg_upsert_method` | Insert/update a Method node |
| `kg_link_paper_method` | Add USES_METHOD edge |
| `kg_get_paper` | Fetch Paper by DOI or arXiv ID |
| `kg_search_fulltext` | Fulltext search over titles + abstracts |
| `kg_search_vector` | Cosine similarity search by embedding |
| `kg_get_topic_papers` | Papers tagged with a topic |
| `kg_get_citation_network` | Citation subgraph up to N hops |
| `kg_get_paper_claims` | All claims from a paper |
| `kg_get_contradictions` | Contradicting claim pairs across KG |
| `kg_stats` | Node and relationship counts |
| `kg_cypher` | Run a read-only Cypher query |

### research_extract

Uses a local LLM via Ollama (OpenAI-compatible API). Default model: `nemotron-cascade-2`.
Configure `base_url`, `model`, and `api_key` under `[extract]` in `config.toml`.
No API key required — run `ollama pull nemotron-cascade-2` to get the model.

| Tool | Description |
|------|-------------|
| `extract_paper_claims` | Extract SPO triples (subject-predicate-object claims) from title+abstract |
| `extract_paper_methods` | Extract algorithms and frameworks mentioned in a paper |
| `extract_paper_results` | Extract quantitative results (metrics, values, units) |
| `extract_paper_full` | Extract claims + methods + results in one Claude call (most efficient) |
| `process_paper` | Fetch paper from KG → extract all → store back (full pipeline) |
| `batch_process_papers` | Process multiple papers concurrently |
| `find_contradictions_for_paper` | Find + LLM-verify contradicting claims for a given paper |

#### Typical workflow

```
1. kg_upsert_paper(paper)           # store paper metadata
2. process_paper(doi)               # extract + store claims/methods/results
3. kg_get_paper_claims(doi)         # inspect what was extracted
4. find_contradictions_for_paper(doi)  # check for contradictions in KG
```

## Knowledge graph schema

**Nodes:** Paper · Author · Claim · Method · Result · Topic

**Edges:** CITES · AUTHORED · ASSERTS · SUPPORTS · CONTRADICTS · USES_METHOD · REPORTS · MEASURES · COVERS

## Configuration

Set `RESEARCH_CONFIG=/path/to/config.toml` to override the default config location.

Override Neo4j credentials via env vars: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.

Add a Semantic Scholar API key in `config.toml` under `[semantic_scholar]` to raise the rate limit from 0.33 req/s to 1 req/s.

Change `[extract] model` in `config.toml` to `claude-haiku-4-5` for bulk claim extraction (~5x cheaper, still high quality for abstracts).
