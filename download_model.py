#!/usr/bin/env python3
"""Télécharge un modèle Hugging Face dans le cache local.

Par défaut, télécharge le modèle enseignant utilisé dans generate.py
(voir DEFAULT_MODEL). Pratique pour pré-remplir le cache sur le serveur GPU
avant de lancer generate.py, notamment quand ce serveur n'a pas d'accès
réseau sortant au moment de l'exécution de vLLM.
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

DEFAULT_MODEL = "Qwen/Qwen3-8B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ID du modèle Hugging Face à télécharger.")
    parser.add_argument("--revision", default=None, help="Révision/branche/tag du modèle (défaut : main).")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Répertoire de cache (défaut : cache Hugging Face standard, $HF_HOME ou ~/.cache/huggingface).",
    )
    args = parser.parse_args()

    print(f"Téléchargement de {args.model} dans le cache Hugging Face...")
    path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    print(f"Modèle téléchargé dans {path}")


if __name__ == "__main__":
    main()
