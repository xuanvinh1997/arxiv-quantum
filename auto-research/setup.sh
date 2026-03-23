#!/usr/bin/env bash
set -euo pipefail

PYTHON=/mnt/hdd/miniconda3/envs/research/bin/python
PIP=/mnt/hdd/miniconda3/envs/research/bin/pip
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing Python dependencies ==="
$PIP install \
  "mcp[cli]>=1.0.0" \
  "fastmcp>=0.4.0" \
  "neo4j>=5.18.0" \
  "feedparser>=6.0.11" \
  "httpx>=0.27.0" \
  "aiofiles>=24.1.0" \
  "tomli>=2.0.2" \
  "pyyaml>=6.0.2" \
  "sentence-transformers>=3.0.0" \
  "openai>=1.0.0"

echo "=== Starting Neo4j ==="
docker compose -f "$SCRIPT_DIR/docker-compose.neo4j.yml" up -d

echo "=== Waiting for Neo4j to be ready (up to 60s) ==="
RETRIES=0
until docker exec research-neo4j cypher-shell -u neo4j -p researchpw "RETURN 1" >/dev/null 2>&1; do
  RETRIES=$((RETRIES + 1))
  if [ $RETRIES -ge 20 ]; then
    echo "ERROR: Neo4j did not start in time"
    exit 1
  fi
  echo "  waiting... ($RETRIES/20)"
  sleep 3
done
echo "  Neo4j is ready!"

echo "=== Initializing schema ==="
$PYTHON "$SCRIPT_DIR/scripts/init_neo4j_schema.py"

echo "=== Seeding KG with existing papers ==="
$PYTHON "$SCRIPT_DIR/scripts/seed_kg.py"

echo ""
echo "=== Setup complete! ==="
echo "  Neo4j browser: http://localhost:7474 (neo4j / researchpw)"
echo "  To test MCP servers:"
echo "    mcp dev $SCRIPT_DIR/servers/research_kg/server.py"
echo "    mcp dev $SCRIPT_DIR/servers/research_fetch/server.py"
echo "    mcp dev $SCRIPT_DIR/servers/research_extract/server.py"
echo "  Note: research_extract requires Ollama running (ollama serve) with nemotron-cascade-2 pulled."
