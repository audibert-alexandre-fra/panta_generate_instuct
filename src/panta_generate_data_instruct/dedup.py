"""Déduplication sémantique par embeddings (sentence-transformers).

Utilisé par generate.py à deux niveaux : réduction d'un pool de candidats surgénérés
au sein d'une cellule (thème x sous-thème x persona), et dédoublonnage final du
dataset à l'intérieur de chaque sous-thème.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_embedder_cache: dict[str, SentenceTransformer] = {}


def get_embedder(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]


def deduplicate(
    texts: list[str],
    threshold: float = 0.85,
    embedder: SentenceTransformer | None = None,
) -> list[int]:
    """Renvoie les indices de `texts` à conserver après dédoublonnage glouton par
    similarité cosinus : parcourt les textes dans l'ordre et rejette tout texte dont
    la similarité avec un texte déjà conservé dépasse `threshold`."""
    if not texts:
        return []
    if len(texts) == 1:
        return [0]

    embedder = embedder or get_embedder()
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    kept: list[int] = []
    kept_embeddings: list[np.ndarray] = []
    for i, emb in enumerate(embeddings):
        if kept_embeddings:
            sims = np.stack(kept_embeddings) @ emb
            if sims.max() >= threshold:
                continue
        kept.append(i)
        kept_embeddings.append(emb)
    return kept
