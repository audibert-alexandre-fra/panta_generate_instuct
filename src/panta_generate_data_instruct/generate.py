"""Génération d'exemples instruct via un modèle enseignant servi par vLLM.

Modèle par défaut : un modèle léger (Qwen 8B) pour les tests. Une fois le pipeline
validé, passer un modèle plus grand (ex. Qwen 32B) via --model.

Procédure par cellule (thème x sous-thème x persona compatible, cf.
SousTheme.poids_personas) :
1. "instruction" et "output" sont générés en deux appels séparés (build_instruction_prompt
   / build_output_prompt) : un appel unique tend à lisser le bruit caractéristique du
   persona dans l'instruction. Le cas (RÉPONSE/RENVOI) de chaque cellule est fixé à
   l'avance par _plan_cells, avec un split RÉPONSE/RENVOI calculé à l'échelle du
   sous-thème (pas persona par persona, cf. _distribuer_renvoi) pour garantir le ratio
   cible même à petit quota.
2. Les instructions sont surgénérées (parametres_generation.surgeneration_min/max) en
   plusieurs vagues ; chaque vague reçoit la liste des instructions déjà retenues pour
   la cellule (consigne : ne pas y ressembler), passe par un troisième appel modèle de
   validation binaire de cohérence sémantique (_filtrer_coherence_semantique, qui
   écarte le charabia avant la génération de l'output), puis un dédoublonnage
   sémantique (dedup.py) réduit le pool avant la vague suivante.
3. Le pool final est réduit au quota de la cellule (dérivé du poids de compatibilité
   persona x sous-thème).
4. Chaque output généré passe par une relecture-réparation (_relire_outputs, appel
   modèle) avant les filtres regex de rejet.
5. Un dernier dédoublonnage sémantique s'applique à l'intérieur de chaque sous-thème,
   toutes personas confondues.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from panta_generate_data_instruct.dedup import deduplicate, get_embedder, similarites_par_paire
from panta_generate_data_instruct.filters import (
    demonstratifs_non_introduits,
    destinataire_absent_de_la_liste,
    deuxieme_personne_designe_tiers,
    engagement_physique_assistant,
    explication_interdite_en_renvoi,
    guillemets_ou_ponctuation_invalides,
    imperatif_delegation,
    incorrections_frequentes,
    interrogatif_incoherent_en_renvoi,
    mot_anglais_isole,
    reparer_elisions,
    reponse_evasive_ou_question,
    structure_sujet_verbe_manquante,
)
from panta_generate_data_instruct.prompts import (
    COHERENCE_SYSTEM_PROMPT,
    RELECTURE_SYSTEM_PROMPT,
    CasType,
    build_coherence_check_prompt,
    build_instruction_prompt,
    build_output_prompt,
    build_relecture_prompt,
    build_system_prompt,
    choisir_format_reponse,
    choisir_variante_renvoi,
)
from panta_generate_data_instruct.schemas import (
    CoherenceCheck,
    GeneratedInstruction,
    GeneratedOutput,
    InstructExample,
    OutputRelecture,
    Persona,
    SousTheme,
    Taxonomy,
    Theme,
)

DEFAULT_MODEL = "Qwen/Qwen3-8B"
N_VAGUES = 3
INSTRUCTION_SCHEMA = GeneratedInstruction.model_json_schema()
OUTPUT_SCHEMA = GeneratedOutput.model_json_schema()
COHERENCE_SCHEMA = CoherenceCheck.model_json_schema()
RELECTURE_SCHEMA = OutputRelecture.model_json_schema()

logger = logging.getLogger(__name__)


def _exige_grammaire_correcte(persona: Persona) -> bool:
    return "correct" in persona.grammaire.lower()


def _filtrer_coherence_semantique(
    llm: LLM,
    candidates: list[Candidate],
    max_tokens: int,
) -> None:
    """Appel de validation binaire au modèle enseignant, entre les deux passes.
    Contrairement au filtre par similarité d'embeddings (qui détecte une dérive
    thématique mais laisse passer un charabia thématiquement proche, ex. "Combien de
    temps pour enfiler la ville ?"), cet appel cible le non-sens véritable. Mute
    `candidate.instruction` à None pour les candidats rejetés (écarté avant la
    génération de l'output). Température fixée à 0 : c'est une vérification, pas une
    génération créative."""
    a_verifier = [c for c in candidates if c.instruction is not None]
    if not a_verifier:
        return

    conversations = [
        [
            {"role": "system", "content": COHERENCE_SYSTEM_PROMPT},
            {"role": "user", "content": build_coherence_check_prompt(c.instruction)},
        ]
        for c in a_verifier
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        structured_outputs=StructuredOutputsParams(json=COHERENCE_SCHEMA),
    )
    outputs = llm.chat(conversations, sampling_params=sampling_params)

    for candidate, output in zip(a_verifier, outputs):
        text = output.outputs[0].text
        tag = f"{candidate.plan.theme.id}/{candidate.plan.sous_theme.id}/{candidate.plan.persona.id}/{candidate.cas}"
        try:
            parsed = CoherenceCheck.model_validate_json(text)
        except ValueError as exc:
            logger.warning("[%s] vérification de cohérence invalide, candidat ignoré : %s", tag, exc)
            candidate.instruction = None
            continue
        if not parsed.sens:
            logger.warning(
                "[%s] instruction rejetée (non-sens détecté par le modèle) : %s",
                tag, candidate.instruction,
            )
            candidate.instruction = None


def _relire_outputs(
    llm: LLM,
    texts: list[str],
    max_tokens: int,
) -> list[str]:
    """Passe de relecture-réparation appliquée à chaque output déjà généré (appel
    modèle, cf. prompts.build_relecture_prompt), après la réparation déterministe des
    élisions et avant les filtres regex de rejet. Corrige les fautes qui échappent à
    reparer_elisions (coquilles, accords, tournures incorrectes ou anglicismes, ex.
    "és" pour "assez", "là-das" pour "là-bas", "au caisse" pour "à la caisse", "plus
    focus") sans changer le sens, le registre ni la longueur. Température fixée à 0 :
    c'est une correction, pas une génération créative. N'échoue jamais un candidat sur
    cette passe : en cas de sortie invalide, le texte d'origine est conservé et seuls
    les filtres suivants peuvent rejeter le candidat."""
    if not texts:
        return texts

    conversations = [
        [
            {"role": "system", "content": RELECTURE_SYSTEM_PROMPT},
            {"role": "user", "content": build_relecture_prompt(text)},
        ]
        for text in texts
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        structured_outputs=StructuredOutputsParams(json=RELECTURE_SCHEMA),
    )
    outputs = llm.chat(conversations, sampling_params=sampling_params)

    corrected: list[str] = []
    for text, output in zip(texts, outputs):
        raw = output.outputs[0].text
        try:
            parsed = OutputRelecture.model_validate_json(raw)
            corrected.append(parsed.output_corrige)
        except ValueError as exc:
            logger.warning("relecture invalide, texte d'origine conservé : %s", exc)
            corrected.append(text)
    return corrected


def build_llm(model: str = DEFAULT_MODEL, enforce_eager: bool = True, **llm_kwargs) -> LLM:
    # enforce_eager=True par défaut : évite torch.compile, dont la passe de fusion
    # AllReduce importe flashinfer.comm, qui utilise l'annotation `array.array[int]`
    # incompatible avec Python < 3.13 (TypeError: 'array.array' is not subscriptable).
    return LLM(model=model, enforce_eager=enforce_eager, **llm_kwargs)


@dataclass
class CellPlan:
    theme: Theme
    sous_theme: SousTheme
    persona: Persona
    cas: CasType
    target: int
    overgen_count: int


@dataclass
class Candidate:
    plan: CellPlan
    intention: str
    exemple_instruction: str
    instruction: str | None = None

    @property
    def cas(self) -> CasType:
        return self.plan.cas


def _distribuer_renvoi(targets: dict[str, int], n_renvoi: int) -> dict[str, int]:
    """Distribue n_renvoi exemples RENVOI entre les personas d'un même sous-thème
    (méthode du plus fort reste, plafonnée par le quota `targets[persona_id]` de
    chaque persona), au lieu d'arrondir indépendamment le split RÉPONSE/RENVOI cellule
    par cellule : à petit quota par persona (ex. target=1), round(1 * ratio_reponse)
    vaut systématiquement 1, ce qui ramenait le RENVOI à zéro dans toutes les cellules
    et faisait disparaître entièrement le cas RENVOI du jeu généré. En calculant le
    nombre de RENVOI une seule fois pour tout le sous-thème (cf. _plan_cells), le ratio
    est respecté à cette échelle même quand aucune cellule individuelle n'a un quota
    assez grand pour en tirer un de son côté."""
    total = sum(targets.values())
    if total <= 0 or n_renvoi <= 0:
        return {persona_id: 0 for persona_id in targets}

    n_renvoi = min(n_renvoi, total)
    raw = {persona_id: n_renvoi * t / total for persona_id, t in targets.items()}
    alloc = {persona_id: min(int(raw[persona_id]), targets[persona_id]) for persona_id in targets}
    restant = n_renvoi - sum(alloc.values())

    # Plus fort reste, en respectant le plafond de chaque persona. Boucle (pas une
    # seule passe) : une fois un plafond atteint, le reliquat doit pouvoir se reporter
    # sur les personas suivants dans l'ordre des plus forts restes.
    ordre = sorted(targets, key=lambda p: raw[p] - int(raw[p]), reverse=True)
    while restant > 0:
        progresse = False
        for persona_id in ordre:
            if restant <= 0:
                break
            if alloc[persona_id] < targets[persona_id]:
                alloc[persona_id] += 1
                restant -= 1
                progresse = True
        if not progresse:
            break

    return alloc


def _plan_cells(
    taxonomy: Taxonomy,
    n_per_cell: int,
    surgeneration_min: int,
    surgeneration_max: int,
) -> list[CellPlan]:
    """Calcule, pour chaque cellule compatible (poids > 0) et pour chaque cas
    (RÉPONSE/RENVOI), le nombre d'exemples cible et le nombre de candidats à
    surgénérer. Le volume cible par sous-thème (n_per_cell x nombre de personas) est
    d'abord redistribué entre les personas compatibles au prorata de leur poids ; les
    personas à poids 0 (ex. personas adultes pour le thème école) sont exclus et leur
    part reportée sur les personas restants. Le split RÉPONSE/RENVOI n'est PAS arrondi
    persona par persona (cf. _distribuer_renvoi pour la raison) : le nombre total de
    RENVOI est calculé une seule fois pour tout le sous-thème
    (max(1, round(total * (1 - ratio_reponse))), garantissant au moins un RENVOI par
    sous-thème dès que son quota total est positif), puis réparti entre les personas
    compatibles au prorata de leur quota. Le cas (et son quota résultant) devient une
    propriété FIXE du plan (plus un tirage aléatoire par candidat), ce qui garantit le
    ratio à l'échelle du sous-thème et pas seulement en moyenne globale sur tout le
    jeu."""
    plans: list[CellPlan] = []
    n_personas = len(taxonomy.personas)
    for theme in taxonomy.themes:
        for sous_theme in theme.sous_themes:
            poids = {p.id: taxonomy.poids_persona(sous_theme, p) for p in taxonomy.personas}
            total_weight = sum(w for w in poids.values() if w > 0)
            if total_weight <= 0:
                logger.warning(
                    "%s/%s : aucun persona compatible, cellule ignorée", theme.id, sous_theme.id
                )
                continue
            total_target = n_per_cell * n_personas
            persona_targets: dict[str, int] = {}
            for persona in taxonomy.personas:
                w = poids[persona.id]
                if w <= 0:
                    continue
                target = round(total_target * w / total_weight)
                if target <= 0:
                    continue
                persona_targets[persona.id] = target

            if not persona_targets:
                continue

            sous_theme_total = sum(persona_targets.values())
            n_renvoi_total = max(1, round(sous_theme_total * (1 - sous_theme.ratio_reponse)))
            n_renvoi_total = min(n_renvoi_total, sous_theme_total)
            renvoi_par_persona = _distribuer_renvoi(persona_targets, n_renvoi_total)

            for persona in taxonomy.personas:
                target = persona_targets.get(persona.id)
                if target is None:
                    continue
                target_renvoi = renvoi_par_persona[persona.id]
                target_reponse = target - target_renvoi
                for cas, cas_target in (("reponse", target_reponse), ("renvoi", target_renvoi)):
                    if cas_target <= 0:
                        continue
                    overgen_count = max(surgeneration_min, min(surgeneration_max, cas_target * 3))
                    plans.append(CellPlan(theme, sous_theme, persona, cas, cas_target, overgen_count))
    return plans


def _generate_instruction_pools(
    llm: LLM,
    taxonomy: Taxonomy,
    plans: list[CellPlan],
    temperature: float,
    max_tokens: int,
) -> list[Candidate]:
    system_prompt = build_system_prompt(taxonomy.style_guide)
    seuil = taxonomy.parametres_generation.dedup_seuil_cosinus
    seuil_exemple = taxonomy.parametres_generation.similarite_exemple_max
    seuil_coherence = taxonomy.parametres_generation.coherence_similarite_min
    embedder = get_embedder()
    pools: dict[int, list[Candidate]] = {id(plan): [] for plan in plans}
    wave_size = {id(plan): max(1, math.ceil(plan.overgen_count / N_VAGUES)) for plan in plans}

    for vague in range(N_VAGUES):
        active_plans = [p for p in plans if len(pools[id(p)]) < p.overgen_count]
        if not active_plans:
            break

        conversations = []
        wave_candidates: list[Candidate] = []
        for plan in active_plans:
            deja_retenues = [c.instruction for c in pools[id(plan)]]
            for _ in range(wave_size[id(plan)]):
                intentions = (
                    plan.sous_theme.intentions_reponse
                    if plan.cas == "reponse"
                    else plan.sous_theme.intentions_renvoi
                )
                intention = random.choice(intentions) if intentions else plan.sous_theme.description
                exemples_theme = plan.persona.exemples_instruction_par_theme.get(plan.theme.id, [])
                exemple_instruction = random.choice(exemples_theme) if exemples_theme else plan.sous_theme.description
                concept = None
                type_question = None
                if plan.cas == "reponse":
                    concept = random.choice(plan.theme.concepts) if plan.theme.concepts else plan.sous_theme.description
                    type_question = (
                        random.choice(taxonomy.types_question)
                        if taxonomy.types_question
                        else "une question de définition"
                    )
                prompt = build_instruction_prompt(
                    plan.theme,
                    plan.sous_theme,
                    plan.persona,
                    plan.cas,
                    intention,
                    exemple_instruction,
                    deja_retenues,
                    concept=concept,
                    type_question=type_question,
                )
                conversations.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                )
                wave_candidates.append(
                    Candidate(
                        plan=plan,
                        intention=intention,
                        exemple_instruction=exemple_instruction,
                    )
                )

        logger.info("Vague %d/%d instructions : %d appels", vague + 1, N_VAGUES, len(conversations))
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            structured_outputs=StructuredOutputsParams(json=INSTRUCTION_SCHEMA),
        )
        outputs = llm.chat(conversations, sampling_params=sampling_params)

        for candidate, output in zip(wave_candidates, outputs):
            text = output.outputs[0].text
            tag = f"{candidate.plan.theme.id}/{candidate.plan.sous_theme.id}/{candidate.plan.persona.id}/{candidate.cas}"
            try:
                parsed = GeneratedInstruction.model_validate_json(text)
            except ValueError as exc:
                logger.warning("[%s] instruction invalide, candidat ignoré : %s", tag, exc)
                continue

            instruction = parsed.instruction

            demonstratifs = demonstratifs_non_introduits(instruction)
            if demonstratifs:
                logger.warning(
                    "[%s] instruction rejetée (démonstratif non introduit %s) : %s",
                    tag, demonstratifs, instruction,
                )
                continue
            if _exige_grammaire_correcte(candidate.plan.persona) and structure_sujet_verbe_manquante(instruction):
                logger.warning(
                    "[%s] instruction rejetée (structure sujet-verbe manquante, "
                    "incompatible avec la grammaire correcte exigée par le persona) : %s",
                    tag, instruction,
                )
                continue
            if deuxieme_personne_designe_tiers(instruction):
                logger.warning(
                    "[%s] instruction rejetée (le \"tu\"/\"vous\" désigne le tiers, "
                    "pas l'assistant) : %s", tag, instruction,
                )
                continue
            if mot_anglais_isole(instruction):
                logger.warning(
                    "[%s] instruction rejetée (mot anglais isolé) : %s", tag, instruction,
                )
                continue

            candidate.instruction = instruction

        wave_candidates = [c for c in wave_candidates if c.instruction is not None]
        if wave_candidates:
            similarites_exemple = similarites_par_paire(
                [c.instruction for c in wave_candidates],
                [c.exemple_instruction for c in wave_candidates],
                embedder=embedder,
            )
            similarites_sujet = similarites_par_paire(
                [c.instruction for c in wave_candidates],
                [f"{c.plan.sous_theme.description} {c.intention}" for c in wave_candidates],
                embedder=embedder,
            )
            for candidate, sim_exemple, sim_sujet in zip(
                wave_candidates, similarites_exemple, similarites_sujet
            ):
                tag = f"{candidate.plan.theme.id}/{candidate.plan.sous_theme.id}/{candidate.plan.persona.id}/{candidate.cas}"
                if sim_exemple >= seuil_exemple:
                    logger.warning(
                        "[%s] instruction rejetée (similarité %.2f avec l'exemple persona) : %s",
                        tag, sim_exemple, candidate.instruction,
                    )
                    candidate.instruction = None
                elif sim_sujet < seuil_coherence:
                    logger.warning(
                        "[%s] instruction rejetée (similarité %.2f avec le sujet demandé, "
                        "probablement incohérente) : %s",
                        tag, sim_sujet, candidate.instruction,
                    )
                    candidate.instruction = None

        _filtrer_coherence_semantique(
            llm, [c for c in wave_candidates if c.instruction is not None], max_tokens
        )

        by_plan: dict[int, list[Candidate]] = {}
        for candidate in wave_candidates:
            if candidate.instruction is not None:
                by_plan.setdefault(id(candidate.plan), []).append(candidate)

        for plan in active_plans:
            combined = pools[id(plan)] + by_plan.get(id(plan), [])
            texts = [c.instruction for c in combined]
            keep_idx = deduplicate(texts, threshold=seuil)
            pools[id(plan)] = [combined[i] for i in keep_idx]

    final: list[Candidate] = []
    for plan in plans:
        kept = pools[id(plan)][: plan.target]
        if len(kept) < plan.target:
            logger.warning(
                "%s/%s/%s : %d/%d instructions retenues après surgénération + dédoublonnage",
                plan.theme.id,
                plan.sous_theme.id,
                plan.persona.id,
                len(kept),
                plan.target,
            )
        final.extend(kept)
    return final


def _generate_outputs(
    llm: LLM,
    taxonomy: Taxonomy,
    candidates: list[Candidate],
    temperature: float,
    max_tokens: int,
) -> tuple[list[InstructExample], list[dict]]:
    system_prompt = build_system_prompt(taxonomy.style_guide)

    conversations = []
    for candidate in candidates:
        variante_renvoi = None
        format_directive = None
        if candidate.cas == "renvoi":
            variante_renvoi = choisir_variante_renvoi(candidate.plan.theme, candidate.plan.persona)
        else:
            format_directive = choisir_format_reponse()
        prompt = build_output_prompt(
            candidate.plan.theme,
            candidate.plan.sous_theme,
            candidate.plan.persona,
            candidate.cas,
            candidate.instruction,
            candidate.exemple_instruction,
            variante_renvoi=variante_renvoi,
            format_directive=format_directive,
        )
        conversations.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

    logger.info("Génération des outputs : %d appels", len(conversations))
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        structured_outputs=StructuredOutputsParams(json=OUTPUT_SCHEMA),
    )
    outputs = llm.chat(conversations, sampling_params=sampling_params)

    # Première passe : parsing et réparation déterministe des élisions. La pertinence
    # intention/output (embeddings, plus coûteuse et nécessitant un lot) est vérifiée
    # dans une deuxième passe séparée, uniquement sur les survivants des filtres regex.
    raw_records: list[dict] = []
    parses_ok: list[tuple[Candidate, dict, str]] = []
    for candidate, conversation, output in zip(candidates, conversations, outputs):
        text = output.outputs[0].text
        tag = f"{candidate.plan.theme.id}/{candidate.plan.sous_theme.id}/{candidate.plan.persona.id}/{candidate.cas}"
        logger.info("[%s] instruction : %s", tag, candidate.instruction)
        logger.info("[%s] sortie brute (output) : %s", tag, text)

        record = {
            "theme": candidate.plan.theme.id,
            "sous_theme": candidate.plan.sous_theme.id,
            "persona_id": candidate.plan.persona.id,
            "type_attendu": candidate.cas,
            "intention": candidate.intention,
            "instruction": candidate.instruction,
            "output_prompt": conversation[1]["content"],
            "raw_output": text,
        }

        try:
            parsed = GeneratedOutput.model_validate_json(text)
        except ValueError as exc:
            logger.error("[%s] output invalide, exemple ignoré : %s", tag, exc)
            record["error"] = str(exc)
            raw_records.append(record)
            continue

        # Réparation déterministe des élisions manquantes (règle orthographique sûre),
        # avant la relecture-réparation par appel modèle ci-dessous.
        output_text = reparer_elisions(parsed.output)
        parses_ok.append((candidate, record, output_text))

    # Relecture-réparation par appel modèle (cf. _relire_outputs) : corrige les fautes
    # qui échappent à reparer_elisions (coquilles, accords, tournures incorrectes),
    # avant les filtres regex de rejet ci-dessous, qui s'appliquent donc au texte
    # relu plutôt qu'à la sortie brute du modèle.
    textes_relus = _relire_outputs(llm, [output_text for _, _, output_text in parses_ok], max_tokens)

    survivants: list[tuple[Candidate, str, dict]] = []
    for (candidate, record, _), output_text in zip(parses_ok, textes_relus):
        tag = f"{candidate.plan.theme.id}/{candidate.plan.sous_theme.id}/{candidate.plan.persona.id}/{candidate.cas}"
        rejet: str | None = None
        demonstratifs = demonstratifs_non_introduits(f"{candidate.instruction} {output_text}")
        if demonstratifs:
            rejet = f"démonstratif non introduit {demonstratifs}"
        elif candidate.cas == "renvoi" and explication_interdite_en_renvoi(output_text):
            rejet = "réassurance/explication interdite dans un output RENVOI"
        elif candidate.cas == "renvoi" and destinataire_absent_de_la_liste(
            output_text, candidate.plan.theme.interlocuteurs_pour(candidate.plan.persona.id)
        ):
            rejet = "destinataire absent de la liste (probablement inventé)"
        elif candidate.cas == "renvoi" and interrogatif_incoherent_en_renvoi(candidate.instruction, output_text):
            rejet = "mot interrogatif de la reprise incohérent avec l'instruction"
        elif candidate.cas == "reponse" and reponse_evasive_ou_question(output_text):
            rejet = "question de clarification ou annonce sans contenu dans un output RÉPONSE"
        elif engagement_physique_assistant(output_text):
            rejet = "engagement physique de l'assistant à la place du tiers"
        elif imperatif_delegation(output_text):
            rejet = "impératif de délégation vers le persona"
        elif guillemets_ou_ponctuation_invalides(output_text):
            rejet = "guillemets non appariés ou ponctuation finale manquante (probable troncature)"
        elif mot_anglais_isole(output_text):
            rejet = "mot anglais isolé"
        else:
            fautes = incorrections_frequentes(output_text)
            if fautes:
                rejet = f"incorrections de français {fautes}"

        if rejet is not None:
            logger.warning("[%s] output rejeté (%s) : %s", tag, rejet, output_text)
            record["error"] = rejet
            raw_records.append(record)
            continue

        survivants.append((candidate, output_text, record))

    # Deuxième passe : pertinence intention/output par similarité d'embeddings, en
    # lot. Détecte une réponse qui reste sur le sujet du sous-thème mais répond à un
    # autre angle que celui tiré (ex. "Comment se passe une IRM ?" répondu par ce que
    # montre une IRM, pas par son déroulé).
    examples: list[InstructExample] = []
    if survivants:
        seuil_pertinence = taxonomy.parametres_generation.pertinence_intention_min
        embedder = get_embedder()
        similarites = similarites_par_paire(
            [output_text for _, output_text, _ in survivants],
            [f"{c.plan.sous_theme.description} {c.intention}" for c, _, _ in survivants],
            embedder=embedder,
        )
        for (candidate, output_text, record), similarite in zip(survivants, similarites):
            tag = f"{candidate.plan.theme.id}/{candidate.plan.sous_theme.id}/{candidate.plan.persona.id}/{candidate.cas}"
            if similarite < seuil_pertinence:
                logger.warning(
                    "[%s] output rejeté (similarité %.2f avec l'intention, "
                    "probablement hors angle) : %s",
                    tag, similarite, output_text,
                )
                record["error"] = "output hors angle par rapport à l'intention tirée"
                raw_records.append(record)
                continue

            record["output"] = output_text
            raw_records.append(record)

            examples.append(
                InstructExample(
                    instruction=candidate.instruction,
                    output=output_text,
                    theme=candidate.plan.theme.id,
                    sous_theme=candidate.plan.sous_theme.id,
                    persona_id=candidate.plan.persona.id,
                    type_attendu=candidate.cas,
                    intention=candidate.intention,
                )
            )

    logger.info("%d/%d exemples valides (output)", len(examples), len(candidates))
    return examples, raw_records


def _dedup_final_par_sous_theme(examples: list[InstructExample], threshold: float) -> list[InstructExample]:
    by_sous_theme: dict[str, list[InstructExample]] = {}
    for example in examples:
        by_sous_theme.setdefault(example.sous_theme, []).append(example)

    kept: list[InstructExample] = []
    for sous_theme_id, group in by_sous_theme.items():
        texts = [example.instruction for example in group]
        keep_idx = deduplicate(texts, threshold=threshold)
        if len(keep_idx) < len(group):
            logger.info(
                "Dédoublonnage final [%s] : %d exemples retirés (quasi-doublons)",
                sous_theme_id,
                len(group) - len(keep_idx),
            )
        kept.extend(group[i] for i in keep_idx)
    return kept


def generate_examples(
    llm: LLM,
    taxonomy: Taxonomy,
    n_per_cell: int,
    temperature: float = 0.9,
    max_tokens: int = 512,
    raw_log_path: Path | None = None,
    surgeneration_min: int | None = None,
    surgeneration_max: int | None = None,
) -> list[InstructExample]:
    params = taxonomy.parametres_generation
    surgeneration_min = surgeneration_min if surgeneration_min is not None else params.surgeneration_min
    surgeneration_max = surgeneration_max if surgeneration_max is not None else params.surgeneration_max

    plans = _plan_cells(taxonomy, n_per_cell, surgeneration_min, surgeneration_max)
    logger.info(
        "Cellules compatibles : %d, exemples cible : %d",
        len(plans),
        sum(plan.target for plan in plans),
    )

    candidates = _generate_instruction_pools(llm, taxonomy, plans, temperature, max_tokens)
    logger.info("%d instructions retenues après surgénération + dédoublonnage", len(candidates))

    examples, raw_records = _generate_outputs(llm, taxonomy, candidates, temperature, max_tokens)

    examples = _dedup_final_par_sous_theme(examples, params.dedup_seuil_cosinus)
    logger.info("%d exemples après dédoublonnage final par sous-thème", len(examples))

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
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help=(
            "Active torch.compile dans vLLM (désactivé par défaut : incompatible "
            "avec flashinfer sous Python < 3.13, cf. build_llm)."
        ),
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("src/panta_generate_data_instruct/config/taxonomy.yaml"),
    )
    parser.add_argument(
        "--n-per-cell",
        type=int,
        default=5,
        help="Exemples cible par combinaison thème x sous-thème, moyenné sur les personas compatibles.",
    )
    parser.add_argument(
        "--surgeneration-min",
        type=int,
        default=None,
        help="Surcharge parametres_generation.surgeneration_min de la taxonomie.",
    )
    parser.add_argument(
        "--surgeneration-max",
        type=int,
        default=None,
        help="Surcharge parametres_generation.surgeneration_max de la taxonomie.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw/generated.jsonl"))
    parser.add_argument(
        "--raw-log",
        type=Path,
        default=Path("data/raw/generated_raw_log.json"),
        help="Fichier JSON de debug : prompts + sorties brutes de chaque appel output.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help=(
            "Longueur de contexte du moteur vLLM (KV cache). Nos prompts et sorties "
            "tiennent largement dans quelques milliers de tokens ; réduire cette "
            "valeur (au lieu de la longueur max du modèle, ex. 40960 pour Qwen3-32B) "
            "évite les erreurs 'KV cache memory' sur GPU à mémoire limitée."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction de la mémoire GPU réservée par vLLM (défaut vLLM : 0.9).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    taxonomy = Taxonomy.from_yaml(args.taxonomy)
    llm_kwargs = {"max_model_len": args.max_model_len}
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    llm = build_llm(args.model, enforce_eager=args.enforce_eager, **llm_kwargs)
    examples = generate_examples(
        llm,
        taxonomy,
        args.n_per_cell,
        args.temperature,
        args.max_tokens,
        raw_log_path=args.raw_log,
        surgeneration_min=args.surgeneration_min,
        surgeneration_max=args.surgeneration_max,
    )
    save_jsonl(examples, args.output)
    print(f"{len(examples)} exemples écrits dans {args.output}")


if __name__ == "__main__":
    main()
