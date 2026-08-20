"""Modèles pydantic : taxonomie (source) et exemples instruct (sortie)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class StyleGuide(BaseModel):
    registre: str
    regles: list[str]


class Persona(BaseModel):
    """Profil linguistique fixe d'un persona, appliqué de façon systématique (pas
    aléatoire) à toutes les instructions générées pour ce persona."""

    id: str
    age: str
    profil: str
    niveau_langage: str
    moyen_caa: str
    longueur_max_mots: int
    grammaire: str
    registre_lexical: str
    bruit_caracteristique: str
    # Registre d'adresse que l'ASSISTANT emploie envers ce persona (2e personne du
    # singulier/pluriel de politesse) : "tu" pour les personas enfant/ado, "vous" pour
    # les personas adultes, afin d'éviter un tutoiement infantilisant d'adultes.
    registre_adresse: Literal["tu", "vous"]
    # Exemples (4-5) illustrant le bruit_caracteristique de ce persona, PAR THÈME
    # (clé = Theme.id) : un exemple à coloration médicale ne doit jamais fuiter dans un
    # exemple de thème famille/école/vie_quotidienne. Un exemple est tiré au sort dans
    # la liste du thème courant à chaque génération pour éviter que le modèle ne
    # s'ancre sur un patron unique.
    exemples_instruction_par_theme: dict[str, list[str]]


class SousTheme(BaseModel):
    id: str
    description: str
    # Proportion (0-1) d'exemples de type B (question de connaissance générale,
    # autosuffisante) à générer pour ce sous-thème ; le reste est du type A (aide à
    # communiquer avec le tiers réel du thème). Cf. build_role_bloc dans prompts.py.
    # Doit être 0 si intentions_role_B est vide (aucun angle de connaissance générale
    # plausible pour ce sous-thème).
    ratio_connaissance: float = 0.0
    # Angles concrets pour le rôle A (situation vécue par le persona) : un angle est
    # tiré au sort par exemple généré pour diversifier le contenu au-delà du seul
    # persona.
    intentions_role_A: list[str] = Field(default_factory=list)
    # Angles concrets pour le rôle B (question de connaissance générale, autosuffisante
    # : même réponse pour n'importe qui, n'importe où, n'importe quand). Doit rester
    # vide si aucun angle de ce sous-thème n'est réellement indépendant du contexte
    # personnel du persona.
    intentions_role_B: list[str] = Field(default_factory=list)
    # Poids de compatibilité persona -> sous-thème (0 = combinaison interdite, ex. un
    # persona adulte pour un sous-thème "école"). Persona absent de ce mapping = poids
    # 1 (compatible, part standard). Cf. Taxonomy.poids_persona.
    poids_personas: dict[str, float] = Field(default_factory=dict)


class Theme(BaseModel):
    id: str
    label: str
    contexte: str
    # Interlocuteur réel (tiers avec qui le persona communique) par persona_id, car le
    # lexique/registre de désignation de l'interlocuteur doit coller à l'âge du
    # persona (ex. "ton enseignant·e" pour un enfant, jamais pertinent pour un adulte).
    interlocuteur_par_persona: dict[str, str]
    contraintes_specifiques: list[str] = Field(default_factory=list)
    quota_exemples: int
    sous_themes: list[SousTheme]

    def interlocuteur_pour(self, persona_id: str) -> str:
        try:
            return self.interlocuteur_par_persona[persona_id]
        except KeyError as exc:
            raise KeyError(
                f"Pas d'interlocuteur défini pour le persona {persona_id!r} dans le "
                f"thème {self.id!r} (persona probablement incompatible avec ce thème)"
            ) from exc


class ParametresGeneration(BaseModel):
    """Tunables pilotant la génération, à ajuster sans toucher au code Python."""

    # Part des exemples de rôle A où l'assistant pose une question de clarification
    # plutôt que de proposer directement 2-3 formulations candidates (comportement
    # par défaut du rôle A).
    clarification_ratio_role_a: float = 0.2
    # Seuil de similarité cosinus (embeddings) au-delà duquel deux instructions sont
    # considérées comme des quasi-doublons.
    dedup_seuil_cosinus: float = 0.85
    # Bornes du nombre de candidats surgénérés par cellule avant dédoublonnage.
    surgeneration_min: int = 12
    surgeneration_max: int = 15
    # Seuil de similarité cosinus (embeddings) au-delà duquel une instruction générée
    # est rejetée pour être trop proche de l'exemple_instruction fourni dans le prompt
    # (évite qu'un exemple de persona ne soit recopié quasiment tel quel).
    similarite_exemple_max: float = 0.88


class Taxonomy(BaseModel):
    style_guide: StyleGuide
    personas: list[Persona]
    themes: list[Theme]
    parametres_generation: ParametresGeneration = Field(default_factory=ParametresGeneration)

    @classmethod
    def from_yaml(cls, path: Path) -> "Taxonomy":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        themes = [{"id": theme_id, **theme} for theme_id, theme in data["themes"].items()]
        return cls(
            style_guide=data["style_guide"],
            personas=data["personas"],
            themes=themes,
            parametres_generation=data.get("parametres_generation", {}),
        )

    def persona(self, persona_id: str) -> Persona:
        for persona in self.personas:
            if persona.id == persona_id:
                return persona
        raise KeyError(f"Persona inconnu : {persona_id}")

    def theme(self, theme_id: str) -> Theme:
        for theme in self.themes:
            if theme.id == theme_id:
                return theme
        raise KeyError(f"Thème inconnu : {theme_id}")

    def poids_persona(self, sous_theme: SousTheme, persona: Persona) -> float:
        """Poids de compatibilité persona -> sous-thème (0 = interdit, défaut 1)."""
        return sous_theme.poids_personas.get(persona.id, 1.0)


class GeneratedInstruction(BaseModel):
    """Sortie brute attendue du modèle enseignant pour l'appel "instruction"."""

    instruction: str


class GeneratedOutput(BaseModel):
    """Sortie brute attendue du modèle enseignant pour l'appel "output"."""

    output: str


class InstructExample(BaseModel):
    """Un exemple du dataset instruct final."""

    instruction: str
    output: str
    theme: str
    sous_theme: str
    persona_id: str
    # A : l'assistant aide à formuler/clarifier un message vers le tiers réel du thème.
    # B : l'assistant répond directement à une question de connaissance générale
    # autosuffisante. Cf. build_role_bloc dans prompts.py.
    type_attendu: Literal["A", "B"]
    # Angle concret tiré au sort dans SousTheme.intentions pour cet exemple.
    intention: str
    # Rôle A uniquement : True si l'output pose une question de clarification plutôt
    # que de proposer directement des formulations candidates (tiré selon
    # ParametresGeneration.clarification_ratio_role_a). None pour le rôle B.
    demande_clarification: bool | None = None
