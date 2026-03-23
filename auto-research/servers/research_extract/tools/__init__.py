"""Ollama-powered extraction of claims, methods, and results from paper abstracts."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore


_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.toml"


def _load_config() -> dict:
    path = Path(os.environ.get("RESEARCH_CONFIG", _CONFIG_PATH))
    with open(path, "rb") as f:
        return tomllib.load(f)


def _get_model() -> str:
    cfg = _load_config()
    return cfg.get("extract", {}).get("model", "nemotron-cascade-2")


def _make_client() -> OpenAI:
    cfg = _load_config()
    extract_cfg = cfg.get("extract", {})
    base_url = os.environ.get(
        "OLLAMA_BASE_URL",
        extract_cfg.get("base_url", "http://localhost:11434/v1"),
    )
    api_key = extract_cfg.get("api_key", "ollama")
    return OpenAI(base_url=base_url, api_key=api_key)


def _parse_response(text: str, model_cls):
    """Parse JSON response text and validate against a Pydantic model."""
    data = json.loads(text)
    return model_cls.model_validate(data)


# ---------------------------------------------------------------------------
# Pydantic output schemas
# ---------------------------------------------------------------------------

ClaimType = Literal["result", "method", "hypothesis", "comparison", "limitation", "general"]
Section = Literal["abstract", "introduction", "methods", "results", "conclusion", "unknown"]
MethodCategory = Literal[
    "ansatz", "optimizer", "error_correction", "noise_mitigation",
    "dataset", "framework", "classical_baseline", "metric", "other"
]


class ClaimOutput(BaseModel):
    text: str = Field(description="Full sentence of the claim as stated or paraphrased from the paper")
    claim_type: ClaimType = Field(description="Category of the claim")
    subject: str = Field(description="What the claim is about (e.g. 'VQE', 'MERA ansatz')")
    predicate: str = Field(description="The relationship or action (e.g. 'achieves', 'outperforms', 'requires')")
    object: str = Field(description="The outcome or comparison target (e.g. '99% fidelity', 'QAOA', 'O(log n) depth')")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence (1.0 = explicit in text, 0.5 = inferred)")
    section: Section = Field(default="abstract", description="Paper section where this claim appears")


class ExtractedClaims(BaseModel):
    claims: list[ClaimOutput]


class MethodOutput(BaseModel):
    name: str = Field(description="Canonical method name (e.g. 'ADAM', 'UCCSD ansatz', 'surface code')")
    category: MethodCategory
    variant: str = Field(default="", description="Specific variant or configuration used")
    description: str = Field(default="", description="One-sentence description of how the method is used in this paper")


class ExtractedMethods(BaseModel):
    methods: list[MethodOutput]


class ResultOutput(BaseModel):
    metric_name: str = Field(description="Name of the metric (e.g. 'accuracy', 'fidelity', 'circuit depth')")
    metric_value: str = Field(description="Numerical value as a string (e.g. '0.987', '12', '≥0.99')")
    metric_unit: str = Field(default="", description="Unit if applicable (e.g. '%', 'qubits', 'gates')")
    context: str = Field(default="", description="Brief context: what dataset/system/conditions")
    method_name: str = Field(default="", description="Which method produced this result (empty if unclear)")


class ExtractedResults(BaseModel):
    results: list[ResultOutput]


class FullExtraction(BaseModel):
    claims: list[ClaimOutput]
    methods: list[MethodOutput]
    results: list[ResultOutput]


# ---------------------------------------------------------------------------
# System prompts (with JSON schema appended for structured output guidance)
# ---------------------------------------------------------------------------

_CLAIMS_SYSTEM = """You are a scientific claim extractor specialized in quantum computing and quantum machine learning research.

Given a paper title and abstract, extract the key scientific claims as subject-predicate-object (SPO) triples.

Guidelines:
- Focus on claims about performance, comparisons, theoretical bounds, and novel contributions
- "subject" = the entity making the claim (method, model, algorithm)
- "predicate" = the relationship verb (achieves, outperforms, reduces, proves, requires)
- "object" = what is claimed (a metric, another method, a bound, a property)
- Set confidence=1.0 for explicit statements, 0.7 for strong implications, 0.5 for inferences
- Extract 3-8 claims per abstract; don't pad with trivial observations
- claim_type options: result (empirical), method (algorithmic contribution), hypothesis (conjecture), comparison (vs baseline), limitation (known failure mode), general"""

_METHODS_SYSTEM = """You are a scientific method extractor specialized in quantum computing and quantum machine learning research.

Given a paper title and abstract, identify all algorithms, models, frameworks, and techniques used or proposed.

Guidelines:
- Include both proposed methods and baselines/comparisons
- Use canonical names (e.g. "ADAM" not "adaptive moment estimation", "VQE" not "variational quantum eigensolver")
- category options: ansatz (quantum circuit structure), optimizer (classical/quantum optimizer), error_correction, noise_mitigation, dataset, framework (software like PennyLane/Qiskit), classical_baseline, metric, other
- Extract 2-8 methods per abstract"""

_RESULTS_SYSTEM = """You are a scientific results extractor specialized in quantum computing and quantum machine learning research.

Given a paper title and abstract, extract all quantitative results (numbers, percentages, bounds, scaling relations).

Guidelines:
- Only extract explicit numbers or formulas, not vague qualitative claims
- metric_value should be the exact value as written (e.g. "98.5", "O(log n)", "≥0.99")
- Include metric_unit when present (%, qubits, parameters, etc.)
- context should identify the experimental setting (dataset, system size, noise model)
- method_name: which method/model produced this result
- Extract 0-6 results per abstract; return empty list if no quantitative results"""

_FULL_SYSTEM = """You are a scientific information extractor specialized in quantum computing and quantum machine learning.

Given a paper title and abstract, extract in one pass:
1. Key scientific claims as SPO triples (3-8 claims)
2. Methods and algorithms used (2-8 methods)
3. Quantitative results reported (0-6 results)

Be precise and faithful to the text. Do not invent information not present in the abstract."""

# Append JSON schemas so the local model knows the expected output shape
_SCHEMA_SUFFIX = "\n\nRespond ONLY with valid JSON matching this schema:\n"

_CLAIMS_SYSTEM_WITH_SCHEMA = (
    _CLAIMS_SYSTEM + _SCHEMA_SUFFIX + json.dumps(ExtractedClaims.model_json_schema(), indent=2)
)
_METHODS_SYSTEM_WITH_SCHEMA = (
    _METHODS_SYSTEM + _SCHEMA_SUFFIX + json.dumps(ExtractedMethods.model_json_schema(), indent=2)
)
_RESULTS_SYSTEM_WITH_SCHEMA = (
    _RESULTS_SYSTEM + _SCHEMA_SUFFIX + json.dumps(ExtractedResults.model_json_schema(), indent=2)
)
_FULL_SYSTEM_WITH_SCHEMA = (
    _FULL_SYSTEM + _SCHEMA_SUFFIX + json.dumps(FullExtraction.model_json_schema(), indent=2)
)


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_claims(
    title: str,
    abstract: str,
    doi: str = "",
    model: str | None = None,
) -> list[dict]:
    """Extract scientific claims from a paper abstract using a local LLM.

    Returns a list of claim dicts ready for kg_add_claim.
    """
    client = _make_client()
    model_id = model or _get_model()

    response = client.chat.completions.create(
        model=model_id,
        max_tokens=4096,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _CLAIMS_SYSTEM_WITH_SCHEMA},
            {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"},
        ],
    )

    parsed: ExtractedClaims = _parse_response(response.choices[0].message.content, ExtractedClaims)
    return [
        {
            "paper_doi": doi,
            "text": c.text,
            "claim_type": c.claim_type,
            "subject": c.subject,
            "predicate": c.predicate,
            "object": c.object,
            "confidence": c.confidence,
            "section": c.section,
        }
        for c in parsed.claims
    ]


def extract_methods(
    title: str,
    abstract: str,
    doi: str = "",
    model: str | None = None,
) -> list[dict]:
    """Extract methods and algorithms from a paper abstract using a local LLM."""
    client = _make_client()
    model_id = model or _get_model()

    response = client.chat.completions.create(
        model=model_id,
        max_tokens=2048,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _METHODS_SYSTEM_WITH_SCHEMA},
            {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"},
        ],
    )

    parsed: ExtractedMethods = _parse_response(response.choices[0].message.content, ExtractedMethods)
    return [
        {
            "paper_doi": doi,
            "name": m.name,
            "category": m.category,
            "variant": m.variant,
            "description": m.description,
        }
        for m in parsed.methods
    ]


def extract_results(
    title: str,
    abstract: str,
    doi: str = "",
    model: str | None = None,
) -> list[dict]:
    """Extract quantitative results from a paper abstract using a local LLM."""
    client = _make_client()
    model_id = model or _get_model()

    response = client.chat.completions.create(
        model=model_id,
        max_tokens=2048,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _RESULTS_SYSTEM_WITH_SCHEMA},
            {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"},
        ],
    )

    parsed: ExtractedResults = _parse_response(response.choices[0].message.content, ExtractedResults)
    return [
        {
            "paper_doi": doi,
            "metric_name": r.metric_name,
            "metric_value": r.metric_value,
            "metric_unit": r.metric_unit,
            "context": r.context,
            "method_name": r.method_name,
        }
        for r in parsed.results
    ]


def extract_full(
    title: str,
    abstract: str,
    doi: str = "",
    model: str | None = None,
) -> dict:
    """Extract claims, methods, and results in a single LLM call.

    More token-efficient than three separate calls.
    Returns {"claims": [...], "methods": [...], "results": [...]}
    """
    client = _make_client()
    model_id = model or _get_model()

    response = client.chat.completions.create(
        model=model_id,
        max_tokens=6144,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _FULL_SYSTEM_WITH_SCHEMA},
            {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"},
        ],
    )

    parsed: FullExtraction = _parse_response(response.choices[0].message.content, FullExtraction)

    claims = [
        {
            "paper_doi": doi,
            "text": c.text,
            "claim_type": c.claim_type,
            "subject": c.subject,
            "predicate": c.predicate,
            "object": c.object,
            "confidence": c.confidence,
            "section": c.section,
        }
        for c in parsed.claims
    ]
    methods = [
        {
            "paper_doi": doi,
            "name": m.name,
            "category": m.category,
            "variant": m.variant,
            "description": m.description,
        }
        for m in parsed.methods
    ]
    results = [
        {
            "paper_doi": doi,
            "metric_name": r.metric_name,
            "metric_value": r.metric_value,
            "metric_unit": r.metric_unit,
            "context": r.context,
            "method_name": r.method_name,
        }
        for r in parsed.results
    ]

    return {"claims": claims, "methods": methods, "results": results}


def detect_contradictions_llm(
    claim_a: dict,
    claim_b: dict,
    model: str | None = None,
) -> dict:
    """Use a local LLM to judge whether two claims contradict each other.

    Returns {"contradicts": bool, "confidence": float, "explanation": str}.
    """

    class ContradictionJudgement(BaseModel):
        contradicts: bool = Field(description="True if the two claims are in direct contradiction")
        confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this judgement")
        explanation: str = Field(description="One-sentence explanation of why they do or do not contradict")

    system = (
        "You are a scientific claim analyser. Judge whether two claims directly contradict each other. "
        "Consider that apparent contradictions may be due to different experimental settings."
        + _SCHEMA_SUFFIX
        + json.dumps(ContradictionJudgement.model_json_schema(), indent=2)
    )

    prompt = (
        f"Claim A: {claim_a.get('text', '')}\n"
        f"  (from paper: {claim_a.get('source_paper_doi', 'unknown')})\n\n"
        f"Claim B: {claim_b.get('text', '')}\n"
        f"  (from paper: {claim_b.get('source_paper_doi', 'unknown')})\n\n"
        "Do these two scientific claims directly contradict each other?"
    )

    client = _make_client()
    model_id = model or _get_model()

    response = client.chat.completions.create(
        model=model_id,
        max_tokens=512,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )

    j: ContradictionJudgement = _parse_response(
        response.choices[0].message.content, ContradictionJudgement
    )
    return {
        "contradicts": j.contradicts,
        "confidence": j.confidence,
        "explanation": j.explanation,
    }
