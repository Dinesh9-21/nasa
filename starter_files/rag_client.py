# rag_client.py
from typing import List, Dict, Optional
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# ------------------ CONFIG ------------------
DEFAULT_CHROMA_DIR = Path("chroma_db")
DEFAULT_COLLECTION_NAME = "nasa_missions"

# ------------------ BACKEND DISCOVERY ------------------

def discover_chroma_backends(chroma_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Discover available ChromaDB backends.
    Returns dict {backend_name: {"display_name": str, "path": Path, "collection": str}}
    """
    chroma_dir = chroma_dir or DEFAULT_CHROMA_DIR
    backends = {}

    if not chroma_dir.exists():
        chroma_dir.mkdir(parents=True)

    backend_name = "local_chroma"
    backends[backend_name] = {
        "display_name": "Local ChromaDB",
        "path": str(chroma_dir),
        "collection": DEFAULT_COLLECTION_NAME
    }
    return backends

# ------------------ RAG SYSTEM INITIALIZATION ------------------

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """
    Initialize Chroma collection for NASA RAG system.
    """
    client = chromadb.Client(chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=chroma_dir
    ))

    if collection_name in [c.name for c in client.list_collections()]:
        collection = client.get_collection(collection_name)
    else:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=None, model_name="text-embedding-3-small"
            )
        )

    return collection

# ------------------ DOCUMENT RETRIEVAL ------------------

def retrieve_documents(collection, query: str, n_results: int = 3) -> Dict[str, List[List]]:
    """
    Perform semantic retrieval from ChromaDB.
    Returns dict with 'documents' and 'metadatas' lists.
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])

        # Ensure lists exist
        if not documents:
            documents = [[]]
        if not metadatas:
            metadatas = [[]]

        return {"documents": documents, "metadatas": metadatas}

    except Exception as e:
        print(f"⚠️ Error during document retrieval: {e}")
        return {"documents": [[]], "metadatas": [[]]}

# ------------------ CONTEXT FORMATTING ------------------

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """
    Combine retrieved documents into a single context string with source attributions.
    """
    context_list = []

    for i, doc in enumerate(documents):
        meta = metadatas[i] if i < len(metadatas) else {}
        source = meta.get("source", f"DOC_{i+1}")
        context_list.append(f"[{source}]\n{doc}")

    return "\n\n".join(context_list)
