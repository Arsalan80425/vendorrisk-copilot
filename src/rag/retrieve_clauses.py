"""Retrieve contract clauses from the local FAISS contract index."""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.rag.ingest_contracts import CHUNKS_PATH, INDEX_PATH, embed_texts, ingest_contracts
from src.utils.schemas import ContractEvidence


def _ensure_index() -> None:
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        ingest_contracts()


def _load_chunks() -> list[dict[str, Any]]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def _vendor_matches(chunk_vendor_name: str, requested_vendor_name: str) -> bool:
    requested = requested_vendor_name.lower().strip()
    candidate = chunk_vendor_name.lower().strip()
    return requested in candidate or candidate in requested


def _search_all(query: str, top_n: int) -> list[tuple[int, float]]:
    import faiss

    index = faiss.read_index(str(INDEX_PATH))
    query_vector = embed_texts([query])
    scores, indices = index.search(query_vector, top_n)
    return [
        (int(index_id), float(score))
        for index_id, score in zip(indices[0], scores[0])
        if int(index_id) >= 0
    ]


def retrieve_contract_clauses(vendor_name: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve source-grounded contract clauses, preferring the requested vendor."""
    _ensure_index()
    chunks = _load_chunks()
    ranked = _search_all(f"{vendor_name} {query}", len(chunks))

    vendor_ranked = [
        (index_id, score)
        for index_id, score in ranked
        if _vendor_matches(chunks[index_id]["vendor_name"], vendor_name)
    ]
    selected = vendor_ranked[:top_k] if vendor_ranked else ranked[:top_k]

    return [
        {
            "text": chunks[index_id]["text"],
            "vendor_name": chunks[index_id]["vendor_name"],
            "clause_type": chunks[index_id]["clause_type"],
            "source": chunks[index_id]["source"],
            "score": round(score, 4),
        }
        for index_id, score in selected
    ]


def generate_contract_evidence_summary(
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


def retrieve_contract_evidence(
    vendor_name: str,
    contract_file: str | None = None,
    risk_factors: list[str] | None = None,
    top_k: int = 4,
) -> list[ContractEvidence]:
    """Compatibility wrapper returning Pydantic evidence objects for the workflow/API."""
    query = " ".join(risk_factors or [])
    clauses = retrieve_contract_clauses(vendor_name=vendor_name, query=query, top_k=top_k)
    if contract_file:
        filtered = [clause for clause in clauses if clause["source"] == contract_file]
        if filtered:
            clauses = filtered
    return [
        ContractEvidence(
            contract_file=str(clause["source"]),
            clause_type=str(clause["clause_type"]),
            text=str(clause["text"]),
            score=float(clause["score"]),
        )
        for clause in clauses
    ]


def main() -> None:
    """CLI entrypoint for contract retrieval."""
    parser = argparse.ArgumentParser(description="Retrieve contract clauses from the local FAISS index.")
    parser.add_argument("--vendor-name", required=True, help="Vendor name or partial vendor name.")
    parser.add_argument("--query", required=True, help="Search query, such as 'SLA compliance renewal'.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of clauses to return.")
    args = parser.parse_args()
    clauses = retrieve_contract_clauses(args.vendor_name, args.query, args.top_k)
    print(json.dumps(clauses, indent=2))


if __name__ == "__main__":
    main()
