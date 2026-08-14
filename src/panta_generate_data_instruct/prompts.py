"""Gabarits de prompt FALC pour la génération d'exemples instruct."""

from __future__ import annotations

from panta_generate_data_instruct.schemas import Persona, SousTheme, StyleGuide, Theme

SYSTEM_PROMPT = """Tu es un générateur de données d'entraînement pour un assistant conversationnel \
destiné à des personnes utilisant la Communication Alternative et Améliorée (CAA).

Registre : {registre}

Règles de style à respecter strictement :
{regles}

Tu réponds uniquement avec un objet JSON valide de la forme :
{{"instruction": "...", "output": "..."}}
Aucun texte hors de cet objet JSON."""


USER_PROMPT_TEMPLATE = """Thème : {theme_label}
Contexte : {theme_contexte}
Sous-thème : {sous_theme_id} — {sous_theme_description}
{contraintes_bloc}
Persona :
- Âge : {persona_age}
- Profil : {persona_profil}
- Niveau de langage : {persona_niveau_langage}
- Moyen de CAA utilisé : {persona_moyen_caa}

Génère un exemple d'échange pour ce persona et ce sous-thème :
- "instruction" : le message tel que le persona le composerait via son moyen de CAA \
(peut être télégraphique ou en mots-clés selon son niveau de langage).
- "output" : la réponse de l'assistant, en français correctement écrit, phrase(s) \
complète(s), simple(s) et adaptée(s) au persona, dans le registre FALC.

Réponds uniquement avec l'objet JSON demandé."""


def build_system_prompt(style_guide: StyleGuide) -> str:
    regles = "\n".join(f"- {regle}" for regle in style_guide.regles)
    return SYSTEM_PROMPT.format(registre=style_guide.registre, regles=regles)


def build_user_prompt(theme: Theme, sous_theme: SousTheme, persona: Persona) -> str:
    contraintes_bloc = ""
    if theme.contraintes_specifiques:
        lignes = "\n".join(f"- {c}" for c in theme.contraintes_specifiques)
        contraintes_bloc = f"Contraintes spécifiques au thème :\n{lignes}\n"

    return USER_PROMPT_TEMPLATE.format(
        theme_label=theme.label,
        theme_contexte=theme.contexte,
        sous_theme_id=sous_theme.id,
        sous_theme_description=sous_theme.description,
        contraintes_bloc=contraintes_bloc,
        persona_age=persona.age,
        persona_profil=persona.profil,
        persona_niveau_langage=persona.niveau_langage,
        persona_moyen_caa=persona.moyen_caa,
    )
