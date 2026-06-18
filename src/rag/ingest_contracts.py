"""Ingest contract clauses into a sentence-transformers + FAISS index."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.config import CONTRACTS_DIR, FAISS_DIR, ensure_directories

INDEX_PATH = FAISS_DIR / "contracts.index"
CHUNKS_PATH = FAISS_DIR / "chunks.json"
METADATA_PATH = FAISS_DIR / "metadata.json"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CLAUSE_KEYWORDS = {
    "payment": ("payment", "invoice", "billing", "fees"),
    "sla": ("sla", "service level", "availability", "service availability"),
    "compliance": ("compliance", "soc2", "soc 2", "insurance", "privacy", "dpa", "audit"),
    "renewal": ("renewal", "renew", "expiration", "contract end", "non-renewal"),
    "termination": ("termination", "terminate", "breach", "cure period"),
}


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    """Load the local sentence-transformers model used for contract embeddings."""
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", DEFAULT_EMBEDDING_MODEL)
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts with sentence-transformers and return normalized float32 vectors."""
    if not texts:
        return np.empty((0, 0), dtype="float32")
    vectors = get_embedding_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def _vendor_name_from_contract(text: str, path: Path) -> str:
    first_line = text.strip().splitlines()[0].strip() if text.strip() else path.stem
    first_line = re.sub(r"\s+(Master Services Agreement|Services Agreement|Agreement)$", "", first_line, flags=re.I)
    return first_line.strip() or path.stem.replace("_", " ").title()


def detect_clause_type(text: str) -> str:
    """Detect a coarse contract clause type from heading and clause text."""
    lowered = text.lower()
    heading = lowered.split(":", 1)[0]
    if "payment" in heading:
        return "payment"
    if "sla" in heading or "service level" in heading:
        return "sla"
    if "compliance" in heading:
        return "compliance"
    if "renewal" in heading:
        return "renewal"
    if "termination" in heading:
        return "termination"
    if "data security" in heading or "escalation" in heading:
        return "general"
    for clause_type, keywords in CLAUSE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return clause_type
    return "general"


def split_contract_into_chunks(contract_file: str, vendor_name: str, text: str) -> list[dict[str, Any]]:
    """Split one contract into clause-level chunks using blank lines and headings."""
    raw_parts = [part.strip() for part in re.split(r"\n\s*\n+", text.strip()) if part.strip()]
    chunks: list[dict[str, Any]] = []
    current_heading = "General"

    for part in raw_parts[1:]:
        lines = [line.strip() for line in part.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) > 1 and len(lines[0].split()) <= 5:
            current_heading = lines[0]
            body = " ".join(lines[1:])
        elif len(lines) == 1 and len(lines[0].split()) <= 5 and not lines[0].endswith("."):
            current_heading = lines[0]
            body = ""
        else:
            body = " ".join(lines)

        chunk_text = _clean_text(f"{current_heading}: {body}" if body else current_heading)
        if not chunk_text:
            continue
        chunks.append(
            {
                "chunk_id": len(chunks),
                "text": chunk_text,
                "vendor_name": vendor_name,
                "contract_file": contract_file,
                "clause_type": detect_clause_type(chunk_text),
                "source": contract_file,
            }
        )
    return chunks


def load_contract_chunks() -> list[dict[str, Any]]:
    """Load and chunk all .txt contracts from data/contracts."""
    chunks: list[dict[str, Any]] = []
    for path in sorted(CONTRACTS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        vendor_name = _vendor_name_from_contract(text, path)
        for chunk in split_contract_into_chunks(path.name, vendor_name, text):
            chunk["chunk_id"] = len(chunks)
            chunks.append(chunk)
    return chunks


def ingest_contracts() -> dict[str, Any]:
    """Build the FAISS contract index and chunk metadata artifacts."""
    ensure_directories()
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_contract_chunks()
    if not chunks:
        raise ValueError(f"No .txt contract files found in {CONTRACTS_DIR}")

    embeddings = embed_texts([chunk["text"] for chunk in chunks])

    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(
        json.dumps(
            {
                "embedding_model": os.getenv("SENTENCE_TRANSFORMER_MODEL", DEFAULT_EMBEDDING_MODEL),
                "index_path": str(INDEX_PATH),
                "chunks_path": str(CHUNKS_PATH),
                "chunk_count": len(chunks),
                "embedding_dimension": int(embeddings.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "contracts": len(list(CONTRACTS_DIR.glob("*.txt"))),
        "chunks": len(chunks),
        "index_path": str(INDEX_PATH),
        "chunks_path": str(CHUNKS_PATH),
    }


def main() -> None:
    """CLI entrypoint for contract ingestion."""
    print(json.dumps(ingest_contracts(), indent=2))


if __name__ == "__main__":
    main()
