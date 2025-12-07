"""Utilities for creating embedding vectors from the show descriptions CSV."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import dotenv
from openai import OpenAI

dotenv.load_dotenv()

def build_description_embeddings(
    csv_path: Union[str, Path],
    *,
    title_column: str = "Title",
    description_column: str = "Description",
    model: str = "text-embedding-3-small",
    batch_size: int = 32,
    limit: Optional[int] = None,
    client: Optional[OpenAI] = None,
) -> Dict[str, Tuple[str, List[float]]]:
    """Read show descriptions, request embeddings, and return a mapping.

    Args:
        csv_path: Absolute or relative path to the CSV file.
        title_column: Column name for show titles (defaults to "Title").
        description_column: Column name to pull text from (defaults to "Description").
        model: OpenAI embedding model identifier.
        batch_size: How many descriptions to send per embeddings.create call.
        limit: Optional cap on the number of rows to process, useful for smoke tests.
        client: Optional preconfigured OpenAI client instance.

    Returns:
        Dict mapping each show title to a tuple of (description, embedding vector).

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If the CSV does not contain required columns or batch_size < 1.
        RuntimeError: If no API key is configured and no client is provided.
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    show_data: List[Tuple[str, str]] = []  # (title, description)
    with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if title_column not in fieldnames:
            raise ValueError(
                f"Column '{title_column}' not found in CSV headers: {fieldnames}"
            )
        if description_column not in fieldnames:
            raise ValueError(
                f"Column '{description_column}' not found in CSV headers: {fieldnames}"
            )

        for row in reader:
            title = (row.get(title_column) or "").strip()
            description = (row.get(description_column) or "").strip()
            if not title or not description:
                continue
            show_data.append((title, description))
            if limit is not None and len(show_data) >= limit:
                break

    if not show_data:
        return {}

    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Provide it via environment or pass a client."
            )
        client = OpenAI(api_key=api_key)

    embeddings: Dict[str, Tuple[str, List[float]]] = {}
    for start in range(0, len(show_data), batch_size):
        chunk = show_data[start : start + batch_size]
        descriptions_only = [desc for _, desc in chunk]
        
        # Batch requests to reduce API latency and costs.
        response = client.embeddings.create(model=model, input=descriptions_only)
        data_items = response.data
        if len(data_items) != len(chunk):
            raise RuntimeError(
                "Embedding response size mismatch. Expected "
                f"{len(chunk)} results but received {len(data_items)}."
            )

        for (title, description), datum in zip(chunk, data_items):
            embeddings[title] = (description, datum.embedding)

    return embeddings
