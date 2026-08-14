"""Modèles pydantic : taxonomie (source) et exemples instruct (sortie)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class StyleGuide(BaseModel):
    registre: str
    regles: list[str]


class Persona(BaseModel):
    id: str
    age: str
    profil: str
    niveau_langage: str
    moyen_caa: str


class SousTheme(BaseModel):
    id: str
    description: str


class Theme(BaseModel):
    id: str
    label: str
    contexte: str
    contraintes_specifiques: list[str] = Field(default_factory=list)
    quota_exemples: int
    sous_themes: list[SousTheme]


class Taxonomy(BaseModel):
    style_guide: StyleGuide
    personas: list[Persona]
    themes: list[Theme]

    @classmethod
    def from_yaml(cls, path: Path) -> "Taxonomy":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        themes = [{"id": theme_id, **theme} for theme_id, theme in data["themes"].items()]
        return cls(style_guide=data["style_guide"], personas=data["personas"], themes=themes)

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


class GeneratedPair(BaseModel):
    """Sortie brute attendue du modèle enseignant (décodage JSON contraint)."""

    instruction: str
    output: str


class InstructExample(BaseModel):
    """Un exemple du dataset instruct final."""

    instruction: str
    output: str
    theme: str
    sous_theme: str
    persona_id: str
