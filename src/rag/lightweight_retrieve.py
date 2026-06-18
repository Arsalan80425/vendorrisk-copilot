"""Keyword-based contract retrieval for lightweight deployments (no FAISS/embeddings)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.config import CONTRACTS_DIR

SECTION_HEADINGS = (
    "Payment Terms",
    "SLA Obligations",
    "Compliance Requirements",
    "Renewal Terms",
    "Termination Rights",
    "Data Security",
    "Escalation Procedure",
)

CLAUSE_TYPE_MAP = {
    "payment terms": "payment",
    "sla obligations": "sla",
    "compliance requirements": "compliance",
    "renewal terms": "renewal",
    "termination rights": "termination",
    "data security": "general",
    "escalation procedure": "general",
}


def _slugify(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _vendor_name_from_contract(text: str, path: Path) -> str:
    first_line = text.strip().splitlines()[0].strip() if text.strip() else path.stem
    first_line = re.sub(
        r"\s+(Master Services Agreement|Services Agreement|Agreement)$",
        "",
        first_line,
        flags=re.I,
    )
    return first_line.strip() or path.stem.replace("_", " ").title()


def _find_contract_path(vendor_name: str) -> Path | None:
    slug = _slugify(vendor_name)
    candidate = CONTRACTS_DIR / f"{slug}_contract.txt"
    if candidate.is_file():
        return candidate
    if not CONTRACTS_DIR.is_dir():
        return None
    lowered = vendor_name.lower()
    for path in CONTRACTS_DIR.glob("*_contract.txt"):
        if slug in path.stem.lower() or lowered.replace(" ", "_") in path.stem.lower():
            return path
    return None


def _split_contract_sections(text: str) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    current_heading = "general"
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped in SECTION_HEADINGS:
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append({"heading": current_heading, "text": body})
            current_heading = stripped
            current_lines = []
        elif stripped:
            current_lines.append(stripped)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({"heading": current_heading, "text": body})

    return sections


def _clause_type(heading: str) -> str:
    return CLAUSE_TYPE_MAP.get(heading.lower(), "general")


def _query_terms(risk_factors: list[str]) -> set[str]:
    terms: set[str] = set()
    for factor in risk_factors:
        for word in re.findall(r"[a-z0-9]+", factor.lower()):
            if len(word) > 2:
                terms.add(word)
    extra = {
        "compliance",
        "sla",
        "invoice",
        "renewal",
        "termination",
        "payment",
        "duplicate",
        "breach",
        "overdue",
        "pending",
    }
    terms.update(extra)
    return terms


def _score_section(text: str, heading: str, terms: set[str]) -> float:
    haystack = f"{heading} {text}".lower()
    if not terms:
        return 0.1
    matches = sum(1 for term in terms if term in haystack)
    return round(matches / len(terms), 4)


def retrieve_contract_clauses_lightweight(
    vendor_name: str,
    risk_factors: list[str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve contract sections by keyword overlap without embeddings or FAISS."""
    contract_path = _find_contract_path(vendor_name)
    if contract_path is None:
        return []

    text = contract_path.read_text(encoding="utf-8")
    resolved_vendor = _vendor_name_from_contract(text, contract_path)
    source = contract_path.name
    terms = _query_terms(risk_factors)
    sections = _split_contract_sections(text)

    scored: list[dict[str, Any]] = []
    for section in sections:
        heading = section["heading"]
        body = section["text"]
        score = _score_section(body, heading, terms)
        scored.append(
            {
                "text": f"{heading}\n{body}" if heading != "general" else body,
                "vendor_name": resolved_vendor,
                "clause_type": _clause_type(heading),
                "source": source,
                "score": score,
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def generate_lightweight_explanation(
    vendor_name: str,
    risk_factors: list[str],
    retrieved_clauses: list[dict[str, Any]],
) -> str:
    """Create a deterministic source-grounded evidence summary without an LLM."""
    if not retrieved_clauses:
        factors = "; ".join(risk_factors) if risk_factors else "the supplied risk factors"
        return f"No contract clauses were retrieved for {vendor_name}. Review {factors} manually."

    factor_text = "; ".join(risk_factors) if risk_factors else "the vendor risk factors"
    clause_groups: dict[str, list[dict[str, Any]]] = {}
    for clause in retrieved_clauses:
        clause_groups.setdefault(str(clause["clause_type"]), []).append(clause)

    evidence_parts: list[str] = []
    for clause_type, clauses in clause_groups.items():
        sources = sorted({str(clause["source"]) for clause in clauses})
        preview = str(clauses[0]["text"])
        if len(preview) > 180:
            preview = preview[:177].rstrip() + "..."
        evidence_parts.append(f"{clause_type}: {preview} Source: {', '.join(sources)}.")

    return (
        f"For {vendor_name}, retrieved contract evidence was compared against: {factor_text}. "
        + " ".join(evidence_parts)
    )
