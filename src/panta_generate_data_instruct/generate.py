"""Génération d'exemples instruct via un modèle enseignant servi par vLLM.

Modèle par défaut : un modèle léger (Qwen 8B) pour les tests. Une fois le pipeline
validé, passer un modèle plus grand (ex. Qwen 32B) via --model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from panta_generate_data_instruct.prompts import build_system_prompt, build_user_prompt
from panta_generate_data_instruct.schemas import (
    GeneratedPair,
    InstructExample,
    Persona,
    SousTheme,
    Taxonomy,
    Theme,
)

DEFAULT_MODEL = "Qwen/Qwen3-8B"
GENERATED_PAIR_SCHEMA = GeneratedPair.model_json_schema()

Cell = tuple[Theme, SousTheme, Persona]

logger = logging.getLogger(__name__)


def build_llm(model: str = DEFAULT_MODEL, **llm_kwargs) -> LLM:
    return LLM(model=model, **llm_kwargs)


def _cells(taxonomy: Taxonomy) -> list[Cell]:
    return [
        (theme, sous_theme, persona)
        for theme in taxonomy.themes
        for sous_theme in theme.sous_themes
        for persona in taxonomy.personas
    ]


def generate_examples(
    llm: LLM,
    taxonomy: Taxonomy,
    n_per_cell: int,
    temperature: float = 0.9,
    max_tokens: int = 512,
    raw_log_path: Path | None = None,
) -> list[InstructExample]:
    system_prompt = build_system_prompt(taxonomy.style_guide)
    cell_per_conversation = [cell for cell in _cells(taxonomy) for _ in range(n_per_cell)]

    conversations = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(theme, sous_theme, persona)},
        ]
        for theme, sous_theme, persona in cell_per_conversation
    ]

    logger.info(
        "Lancement de la génération : %d exemples (%d cellules x %d)",
        len(conversations),
        len(cell_per_conversation) // max(n_per_cell, 1),
        n_per_cell,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        structured_outputs=StructuredOutputsParams(json=GENERATED_PAIR_SCHEMA),
    )

    outputs = llm.chat(conversations, sampling_params=sampling_params)

    examples = []
    raw_records = []
    for (theme, sous_theme, persona), conversation, output in zip(
        cell_per_conversation, conversations, outputs
    ):
        text = output.outputs[0].text
        tag = f"{theme.id}/{sous_theme.id}/{persona.id}"
        logger.info("[%s] prompt utilisateur : %s", tag, conversation[1]["content"])
        logger.info("[%s] sortie brute : %s", tag, text)

        record = {
            "theme": theme.id,
            "sous_theme": sous_theme.id,
            "persona_id": persona.id,
            "system_prompt": conversation[0]["content"],
            "user_prompt": conversation[1]["content"],
            "raw_output": text,
        }

        try:
            pair = GeneratedPair.model_validate_json(text)
        except ValueError as exc:
            logger.error("[%s] sortie invalide, exemple ignoré : %s", tag, exc)
            record["error"] = str(exc)
            raw_records.append(record)
            continue

        record["instruction"] = pair.instruction
        record["output"] = pair.output
        raw_records.append(record)

        examples.append(
            InstructExample(
                instruction=pair.instruction,
                output=pair.output,
                theme=theme.id,
                sous_theme=sous_theme.id,
                persona_id=persona.id,
            )
        )

    logger.info("%d/%d exemples valides", len(examples), len(conversations))

    if raw_log_path is not None:
        save_raw_log(raw_records, raw_log_path)
        logger.info("Log brut (prompts + sorties) écrit dans %s", raw_log_path)

    return examples


def save_raw_log(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def save_jsonl(examples: list[InstructExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(example.model_dump_json() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modèle enseignant servi par vLLM.")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("src/panta_generate_data_instruct/config/taxonomy.yaml"),
    )
    parser.add_argument(
        "--n-per-cell",
        type=int,
        default=5,
        help="Exemples générés par combinaison thème x sous-thème x persona.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw/generated.jsonl"))
    parser.add_argument(
        "--raw-log",
        type=Path,
        default=Path("data/raw/generated_raw_log.json"),
        help="Fichier JSON de debug : prompts + sorties brutes de chaque appel.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    taxonomy = Taxonomy.from_yaml(args.taxonomy)
    llm = build_llm(args.model)
    examples = generate_examples(
        llm,
        taxonomy,
        args.n_per_cell,
        args.temperature,
        args.max_tokens,
        raw_log_path=args.raw_log,
    )
    save_jsonl(examples, args.output)
    print(f"{len(examples)} exemples écrits dans {args.output}")


if __name__ == "__main__":
    main()
