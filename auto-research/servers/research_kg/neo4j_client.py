"""Async Neo4j driver wrapper with connection pooling."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

from neo4j import AsyncGraphDatabase, AsyncDriver


_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.toml"


def _load_config() -> dict:
    path = Path(os.environ.get("RESEARCH_CONFIG", _CONFIG_PATH))
    with open(path, "rb") as f:
        return tomllib.load(f)


class Neo4jClient:
    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    @classmethod
    def from_config(cls) -> "Neo4jClient":
        cfg = _load_config()["neo4j"]
        uri = os.environ.get("NEO4J_URI", cfg["uri"])
        user = os.environ.get("NEO4J_USER", cfg["user"])
        password = os.environ.get("NEO4J_PASSWORD", cfg["password"])
        max_pool = cfg.get("max_connection_pool_size", 10)
        driver = AsyncGraphDatabase.driver(
            uri, auth=(user, password), max_connection_pool_size=max_pool
        )
        return cls(driver)

    async def run_query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute a read query and return list of record dicts."""
        async with self._driver.session() as session:
            result = await session.run(cypher, params or {})
            records = await result.data()
            return records

    async def run_write(self, cypher: str, params: dict | None = None) -> dict:
        """Execute a write query and return summary counters."""
        async with self._driver.session() as session:
            result = await session.run(cypher, params or {})
            summary = await result.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "relationships_created": summary.counters.relationships_created,
                "properties_set": summary.counters.properties_set,
            }

    async def run_write_many(self, queries: list[tuple[str, dict]]) -> dict:
        """Execute multiple write queries in a single transaction."""
        totals: dict[str, int] = {
            "nodes_created": 0,
            "relationships_created": 0,
            "properties_set": 0,
        }
        async with self._driver.session() as session:
            async with await session.begin_transaction() as tx:
                for cypher, params in queries:
                    result = await tx.run(cypher, params)
                    summary = await result.consume()
                    totals["nodes_created"] += summary.counters.nodes_created
                    totals["relationships_created"] += summary.counters.relationships_created
                    totals["properties_set"] += summary.counters.properties_set
                await tx.commit()
        return totals

    async def close(self) -> None:
        await self._driver.close()
