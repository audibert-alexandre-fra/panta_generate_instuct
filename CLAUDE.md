# panta_generate_data_instruct

## Objectif

Créer un jeu de données **instruct en français** destiné à l'entraînement (fine-tuning)
d'un **modèle multimodal**, ciblant des **patients utilisateurs de la CAA**
(Communication Alternative et Améliorée).

- Génération via un modèle enseignant servi par **vLLM**.
- Le champ `output` doit toujours être en **français correctement écrit** (grammaire,
  orthographe), même si le champ `instruction` peut être télégraphique (reflet d'un
  message composé via un système de CAA).
- Registre **FALC** (Facile à Lire et à Comprendre) : phrases courtes, vocabulaire
  concret, pas de jargon ni d'expressions idiomatiques, ton direct et bienveillant sans
  être infantilisant.
- Taxonomie autour de **4 thèmes** — médical, famille, vie quotidienne, école — avec
  sous-thèmes, personas et quotas définis dans
  `src/panta_generate_data_instruct/config/taxonomy.yaml` (29 sous-thèmes — 8 pour le
  médical, 7 pour chacun des trois autres thèmes — × 3 personas × `n_per_cell`
  exemples). Pilote initial : 255 exemples (17 sous-thèmes × 3 × 5). La taxonomie a
  depuis été étoffée (29 sous-thèmes, plus d'intentions/concepts/types de question) en
  préparation d'un passage à l'échelle vers ~20 000 exemples (`n_per_cell` ≈ 230),
  pour donner assez de matière combinatoire avant un sur-échantillonnage aussi fort.
- Personas : 3 personas organisés par **tranche d'âge et niveau de langage/vocabulaire**
  (enfant, adolescent, adulte), jamais par profil médical ou type de handicap précis.
- Chaque `instruction` et chaque `output` tiennent toujours en **une seule phrase**
  (filtre de rejet dédié, cf. `filters.plusieurs_phrases`).
- `ratio_reponse` (proportion de cas RÉPONSE vs RENVOI, par sous-thème) fixé à **0.95**
  partout, y compris sur le thème médical (5 % de RENVOI, uniforme sur les 4 thèmes).

## État actuel

- [x] `pyproject.toml` — dépendances (vllm, pydantic, sentence-transformers, pyyaml, tqdm)
- [x] `taxonomy.yaml` — thèmes, sous-thèmes, personas, quotas, guide de style FALC
      (bug de syntaxe corrigé : `contexte` du thème école avait un `:` non échappé)
- [x] `schemas.py` — modèles pydantic : `Taxonomy`, `Persona`, `Theme`, `SousTheme`,
      `GeneratedPair` (sortie brute du modèle), `InstructExample` (exemple final)
- [x] `prompts.py` — gabarits de prompt FALC (system + user) par thème/sous-thème/persona
- [x] `generate.py` — génération via vLLM, sortie JSON contrainte
      (`SamplingParams(structured_outputs=StructuredOutputsParams(json=...))`, API vLLM
      0.27), logging détaillé (prompt + sortie brute par exemple), sauvegarde du
      dataset en JSONL + log brut complet en JSON. Modèle par défaut : `Qwen/Qwen3-8B`
      (léger, pour les tests), à remplacer par `Qwen/Qwen3-32B` une fois le pipeline
      validé. **Pas encore testé avec un vrai modèle** : la machine de dev a un driver
      CUDA trop ancien (11.2) pour cette version de vLLM/PyTorch ; le test réel se fait
      sur un serveur GPU dédié (commande donnée à l'utilisateur, résultat en attente).
- [ ] `judge.py` — filtrage qualité / sécurité (respect des contraintes, ex. pas de
      diagnostic médical dans le thème "médical")
- [ ] `dedup.py` — déduplication par embeddings (sentence-transformers)
- [ ] `pipeline.py` — CLI d'orchestration de bout en bout
- [ ] Test de `generate.py` sur serveur GPU (petit run, `--n-per-cell 1`, 96 exemples)
- [ ] Génération du pilote (~255 exemples, 4 thèmes)
- [ ] Revue qualité manuelle d'un échantillon du pilote
- [ ] Export du dataset final dans `data/processed/`

## TODO

1. ~~**schemas.py**~~ — fait.
2. ~~**prompts.py**~~ — fait.
3. ~~**generate.py**~~ — fait, en attente de validation sur GPU réel.
4. Lancer le test `generate.py` sur le serveur GPU (voir commande ci-dessous) et
   ajuster si besoin (prompt, temperature, max_tokens, parsing JSON).
5. **judge.py** : filtre qualité/sécurité — validation FALC, respect des contraintes
   par thème (ex. médical : jamais de diagnostic/posologie), rejet des sorties non
   conformes.
6. **dedup.py** : déduplication sémantique par embeddings pour éviter les exemples
   trop similaires au sein d'un même sous-thème/persona.
7. **pipeline.py** : CLI orchestrant taxonomie → génération → judge → dedup → export.
8. Lancer la génération pilote complète sur les 4 thèmes (`--n-per-cell 5`, ~255 exemples,
   déjà configuré dans `generate.slurm`),
   d'abord avec Qwen3-8B, puis Qwen3-32B une fois le pipeline validé.
9. Relire manuellement un échantillon (qualité du français, respect FALC, pertinence
   par persona) avant de considérer le pilote validé.
10. Exporter le dataset final validé dans `data/processed/`.

### Commande de test sur serveur GPU

```bash
uv sync
uv run python -m panta_generate_data_instruct.generate \
  --model Qwen/Qwen3-8B \
  --n-per-cell 1 \
  --output data/raw/test_generated.jsonl \
  --raw-log data/raw/test_generated_raw_log.json \
  --log-level INFO
```
