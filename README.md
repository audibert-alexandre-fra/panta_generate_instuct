# panta_generate_data_instruct

Génération synthétique d'un jeu de données instruct pour un assistant conversationnel destiné
aux personnes utilisant la Communication Alternative et Améliorée (CAA), sur quatre thèmes :
**médical** (questions d'un patient à un·e spécialiste), **famille**, **vie quotidienne**, **école**.

Le dataset est produit via un modèle enseignant servi par [vLLM](https://github.com/vllm-project/vllm),
piloté avec [uv](https://docs.astral.sh/uv/).

## État actuel

Projet en cours de construction. Voir `src/panta_generate_data_instruct/config/taxonomy.yaml`
pour les thèmes, sous-thèmes et personas couverts par le pilote (200 à 500 exemples).

## Installation

```bash
uv sync
```

Les dépendances (vllm, sentence-transformers, ...) sont volumineuses : à lancer sur la machine
qui servira réellement le modèle enseignant.

## Structure

```
src/panta_generate_data_instruct/
    config/taxonomy.yaml   # thèmes, sous-thèmes, personas, quotas
    prompts.py              # style FALC + gabarits de prompt par thème
    schemas.py               # modèles pydantic (Persona, InstructExample, ...)
    generate.py                # génération via vLLM (sortie JSON structurée)
    judge.py                     # filtrage qualité / sécurité
    dedup.py                       # déduplication par embeddings
    pipeline.py                     # CLI d'orchestration
data/
    raw/         # sorties brutes de génération (gitignored)
    processed/   # dataset final (gitignored)
```
