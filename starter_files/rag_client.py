import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends: Dict[str, Dict[str, str]] = {}
    current_dir = Path(".")

    # Look for directories that likely contain ChromaDB data
    candidate_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]

    for directory in candidate_dirs:
        try:
            # Attempt to connect to ChromaDB directory
            client = chromadb.Client(
                Settings(
                    persist_directory=str(directory),
                    anonymized_telemetry=False
                )
            )

            collections = client.list_collections()

            for collection in collections:
                key = f"{directory.name}/{collection.name}"

                # Safely get document count
                try:
                    count = collection.count()
                except Exception:
                    count = "unknown"

                backends[key] = {
                    "path": str(directory),
                    "collection": collection.name,
                    "display_name": f"{directory.name} → {collection.name} ({count} docs)",
                    "document_count": count
                }

        except Exception as e:
            # Graceful fallback for unreadable directories
            error_msg = str(e)
            truncated_error = (
                error_msg[:60] + "..."
                if len(error_msg) > 60
                else error_msg
            )

            backends[directory.name] = {
                "path": str(directory),
                "collection": "N/A",
                "display_name": f"{directory.name} (error: {truncated_error})",
                "document_count": "N/A"
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    client = chromadb.Client(
        Settings(
            persist_directory=chroma_dir,
            anonymized_telemetry=False
        )
    )

    return client.get_collection(collection_name)


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    where_filter = None

    if mission_filter and mission_filter.lower() not in {"all", "any"}:
        where_filter = {"mission": mission_filter}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context with explicit DOC_IDs"""
    if not documents:
        return ""

    context_parts = ["### Retrieved NASA Context\n"]

    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        doc_id = (
            meta.get("doc_id")
            or meta.get("source")
            or f"DOC_{idx}"
        )
        mission = meta.get("mission", "unknown mission").replace("_", " ").title()
        category = meta.get("category", "general").replace("_", " ").title()

        header = (
            f"[Source {idx}] "
            f"Mission: {mission} | "
            f"Category: {category} | "
            f"Source: {source}"
        )

        context_parts.append(header)

        # Truncate overly long documents
        max_length = 1200
        if len(doc) > max_length:
            doc = doc[:max_length].rstrip() + "..."

        context_parts.append(doc)
        context_parts.append("")  # spacer line

    return "\n".join(context_parts)
