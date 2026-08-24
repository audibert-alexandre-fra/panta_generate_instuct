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
    # Proportion (0-1) d'exemples du cas RÉPONSE (l'assistant répond directement, de
    # façon factuelle) à générer pour ce sous-thème ; le reste est du cas RENVOI
    # (l'assistant indique qu'il ne peut pas répondre et vers qui se tourner). Le
    # critère de bascule n'est jamais le sujet mais la source de la réponse : dépend-elle
    # de cette personne, de ce lieu, de ce moment ? Cf. les blocs CAS_* dans prompts.py.
    ratio_reponse: float = 0.0
    # Angles concrets pour le cas RÉPONSE (même réponse pour n'importe qui, n'importe
    # où, n'importe quand) : un angle est tiré au sort par exemple généré.
    intentions_reponse: list[str] = Field(default_factory=list)
    # Angles concrets pour le cas RENVOI (la réponse dépend de cette personne, ce lieu
    # ou ce moment précis, ex. un symptôme ressenti, un lieu, un retard, la réaction
    # d'un proche donné).
    intentions_renvoi: list[str] = Field(default_factory=list)
    # Poids de compatibilité persona -> sous-thème (0 = combinaison interdite, ex. un
    # persona adulte pour un sous-thème "école"). Persona absent de ce mapping = poids
    # 1 (compatible, part standard). Cf. Taxonomy.poids_persona.
    poids_personas: dict[str, float] = Field(default_factory=dict)


class Theme(BaseModel):
    id: str
    label: str
    contexte: str
    # Destinataires réels possibles (tiers avec qui le persona communique) par
    # persona_id, sous forme de LISTE (jamais une chaîne énumérative du type "votre
    # médecin ou kinésithérapeute" ou "votre famille (parents, fratrie, conjoint·e)" :
    # une telle chaîne est insérée verbatim dans le prompt et ressort telle quelle,
    # parenthèses comprises, dans l'output). Le modèle choisit lui-même, dans cette
    # liste, le destinataire le plus pertinent pour la question précise (cf.
    # CAS_RENVOI_OUTPUT_BLOC dans prompts.py) ; le lexique de désignation doit coller
    # à l'âge du persona (ex. "ton enseignant·e" pour un enfant, jamais pour un adulte).
    interlocuteur_par_persona: dict[str, list[str]]
    # Amorces lexicales concrètes (objets, phénomènes, lieux, activités du quotidien
    # propres à ce thème) : un concept est tiré au sort par exemple du cas RÉPONSE pour
    # ancrer la question dans le monde plutôt que dans un commentaire méta sur le
    # sous-thème (ex. "un aimant" plutôt que "pourquoi on s'entraide"). Cf.
    # CAS_REPONSE_INSTRUCTION_BLOC dans prompts.py.
    concepts: list[str] = Field(default_factory=list)
    contraintes_specifiques: list[str] = Field(default_factory=list)
    quota_exemples: int
    sous_themes: list[SousTheme]

    def interlocuteurs_pour(self, persona_id: str) -> list[str]:
        try:
            return self.interlocuteur_par_persona[persona_id]
        except KeyError as exc:
            raise KeyError(
                f"Pas d'interlocuteur défini pour le persona {persona_id!r} dans le "
                f"thème {self.id!r} (persona probablement incompatible avec ce thème)"
            ) from exc


class ParametresGeneration(BaseModel):
    """Tunables pilotant la génération, à ajuster sans toucher au code Python."""

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
    # Seuil de similarité cosinus en-deçà duquel une instruction générée est rejetée
    # pour être trop éloignée du sous-thème/de l'intention demandés (détecte une
    # dérive thématique ; ne détecte PAS un charabia qui reste proche du sujet en
    # surface, cf. la vérification par appel modèle dans generate._filtrer_coherence_semantique).
    coherence_similarite_min: float = 0.20
    # Seuil de similarité cosinus en-deçà duquel un output généré est rejeté pour être
    # trop éloigné du sous-thème/de l'intention demandés (détecte une réponse hors
    # sujet par rapport à l'angle tiré, ex. une explication de ce que montre une IRM
    # alors que l'intention portait sur le déroulé de l'examen).
    pertinence_intention_min: float = 0.15


class Taxonomy(BaseModel):
    style_guide: StyleGuide
    personas: list[Persona]
    themes: list[Theme]
    # Formes rhétoriques pour une question du cas RÉPONSE (définition, cause,
    # procédure...), tirée au sort au même titre que l'intention et le concept. Liste
    # globale (pas par thème) : ces formes sont indépendantes du domaine. Le type
    # "durée" est délibérément absent : il produit systématiquement des valeurs
    # numériques inventées (cf. CAS_REPONSE_OUTPUT_BLOC dans prompts.py).
    types_question: list[str] = Field(default_factory=list)
    parametres_generation: ParametresGeneration = Field(default_factory=ParametresGeneration)

    @classmethod
    def from_yaml(cls, path: Path) -> "Taxonomy":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        themes = [{"id": theme_id, **theme} for theme_id, theme in data["themes"].items()]
        return cls(
            style_guide=data["style_guide"],
            personas=data["personas"],
            themes=themes,
            types_question=data.get("types_question", []),
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


class CoherenceCheck(BaseModel):
    """Sortie brute attendue du modèle enseignant pour l'appel de validation binaire
    de correction grammaticale et de cohérence sémantique d'une instruction, entre la
    génération de l'instruction et celle de l'output (cf.
    generate._filtrer_coherence_semantique)."""

    sens: bool


class OutputRelecture(BaseModel):
    """Sortie brute attendue du modèle enseignant pour l'appel de relecture-réparation
    d'un output déjà généré (cf. generate._relire_outputs) : corrige les fautes
    d'orthographe, de grammaire, d'accord ou de frappe sans changer le sens, le
    registre, la longueur ni le niveau de vocabulaire du texte."""

    output_corrige: str


class InstructExample(BaseModel):
    """Un exemple du dataset instruct final."""

    instruction: str
    output: str
    theme: str
    sous_theme: str
    persona_id: str
    # reponse : l'assistant répond directement, de façon factuelle (la réponse est la
    # même pour n'importe qui, n'importe où, n'importe quand).
    # renvoi : l'assistant n'est pas en position de savoir (la réponse dépend de cette
    # personne, ce lieu ou ce moment précis) ; il le dit et indique vers qui se
    # tourner. Cf. les blocs CAS_* dans prompts.py.
    type_attendu: Literal["reponse", "renvoi"]
    # Angle concret tiré au sort dans SousTheme.intentions_reponse/intentions_renvoi
    # pour cet exemple.
    intention: str
