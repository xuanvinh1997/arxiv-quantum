#!/usr/bin/env python3
"""Initialize Neo4j schema: constraints, indices, fulltext, and vector index."""
import asyncio
import sys
from pathlib import Path

# Allow running from project root or scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from servers.research_kg.neo4j_client import Neo4jClient
from servers.research_kg.schema import (
    CONSTRAINTS,
    INDICES,
    FULLTEXT_INDICES,
    VECTOR_INDEX,
)


async def main() -> None:
    client = Neo4jClient.from_config()
    all_statements = CONSTRAINTS + INDICES + FULLTEXT_INDICES + [VECTOR_INDEX.strip()]
    print(f"Applying {len(all_statements)} schema statements...")

    ok = 0
    errors = []
    for stmt in all_statements:
        try:
            await client.run_write(stmt)
            print(f"  OK  {stmt[:70]}")
            ok += 1
        except Exception as exc:
            msg = str(exc)
            print(f"  ERR {stmt[:70]}  →  {msg[:80]}")
            errors.append(msg)

    await client.close()

    print(f"\n{ok}/{len(all_statements)} statements applied.")
    if errors:
        print(f"{len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("Schema ready.")


if __name__ == "__main__":
    asyncio.run(main())
