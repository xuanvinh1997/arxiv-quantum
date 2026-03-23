"""Neo4j schema: constraints, indices, and fulltext index definitions."""

CONSTRAINTS = [
    "CREATE CONSTRAINT paper_doi_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE",
    "CREATE CONSTRAINT claim_id_unique IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT method_name_unique IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT author_s2id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.s2_author_id IS UNIQUE",
    "CREATE CONSTRAINT topic_name_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
]

INDICES = [
    "CREATE INDEX paper_arxiv_idx IF NOT EXISTS FOR (p:Paper) ON (p.arxiv_id)",
    "CREATE INDEX paper_date_idx IF NOT EXISTS FOR (p:Paper) ON (p.published_date)",
    "CREATE INDEX claim_type_idx IF NOT EXISTS FOR (c:Claim) ON (c.claim_type)",
    "CREATE INDEX method_category_idx IF NOT EXISTS FOR (m:Method) ON (m.category)",
]

FULLTEXT_INDICES = [
    """CREATE FULLTEXT INDEX paper_text_idx IF NOT EXISTS
       FOR (p:Paper) ON EACH [p.title, p.abstract]""",
    """CREATE FULLTEXT INDEX claim_text_idx IF NOT EXISTS
       FOR (c:Claim) ON EACH [c.text, c.subject, c.object]""",
]

VECTOR_INDEX = """
CREATE VECTOR INDEX paper_embedding_idx IF NOT EXISTS
FOR (p:Paper) ON (p.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}}
"""

# Schema documentation (for reference only, not executed)
SCHEMA_DOCS = {
    "nodes": {
        "Paper": "doi (PK), arxiv_id, title, abstract, published_date, updated_date, pdf_url, categories[], embedding[]",
        "Author": "s2_author_id (PK), name, affiliation",
        "Claim": "id (PK), text, claim_type, subject, predicate, object, confidence, source_paper_doi",
        "Method": "name (PK), category, description",
        "Result": "id, metric_name, metric_value, metric_unit, context, source_paper_doi",
        "Topic": "name (PK), canonical_form",
    },
    "relationships": {
        "CITES": "(Paper)->(Paper), context",
        "AUTHORED": "(Author)->(Paper), position",
        "ASSERTS": "(Paper)->(Claim), section",
        "SUPPORTS": "(Claim)->(Claim), strength",
        "CONTRADICTS": "(Claim)->(Claim), explanation, confidence",
        "USES_METHOD": "(Paper)->(Method), variant, configuration",
        "REPORTS": "(Paper)->(Result)",
        "MEASURES": "(Result)->(Method)",
        "COVERS": "(Paper)->(Topic), (Claim)->(Topic)",
    },
}
