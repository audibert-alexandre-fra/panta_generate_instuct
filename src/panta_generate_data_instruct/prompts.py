"""Gabarits de prompt FALC pour la génération d'exemples instruct."""

from __future__ import annotations

from panta_generate_data_instruct.schemas import Persona, SousTheme, StyleGuide, Theme

SYSTEM_PROMPT = """Tu es un générateur de données d'entraînement pour un modèle destiné à des \
personnes utilisant la Communication Alternative et Améliorée (CAA).

Chaque exemple simule un échange réel : "instruction" est le message composé par le \
persona via son moyen de CAA, et "output" est la réponse que donnerait réellement \
l'interlocuteur·rice du contexte (le·la professionnel·le, proche ou camarade concerné·e \
— précisé dans chaque prompt utilisateur), pas une réponse générique d'assistant. \
Cet·te interlocuteur·rice s'exprime dans un français simple et accessible, adapté à une \
personne qui communique via la CAA.

Registre : {registre}

Règles de style à respecter strictement :
{regles}

Tu réponds uniquement avec un objet JSON valide de la forme :
{{"instruction": "...", "output": "..."}}
Aucun texte hors de cet objet JSON."""


USER_PROMPT_TEMPLATE = """Thème : {theme_label}
Contexte : {theme_contexte}
Interlocuteur·rice à qui le persona s'adresse : {theme_interlocuteur}
Sous-thème : {sous_theme_id} — {sous_theme_description}
{contraintes_bloc}
Persona :
- Âge : {persona_age}
- Profil : {persona_profil}
- Niveau de langage : {persona_niveau_langage}
- Moyen de CAA utilisé : {persona_moyen_caa}
- Exemple de niveau de langage pour ce persona (à adapter au sous-thème, ne pas \
recopier tel quel) : "{persona_exemple_instruction}"

Génère un exemple d'échange pour ce persona et ce sous-thème :
- "instruction" : le message tel que le persona le composerait via son moyen de CAA, \
dans le même registre que l'exemple de niveau de langage ci-dessus (ni plus \
télégraphique, ni plus élaboré). Même simplifié, il doit toujours rester une phrase \
à peu près correcte et reconnaissable comme une phrase (mots manquants ou \
conjugaison approximative tolérés), portant une vraie question ou une vraie demande \
avec une intention claire, compréhensible sans contexte supplémentaire. Une simple \
liste de mots-clés juxtaposés sans lien grammatical n'est jamais acceptable, quel que \
soit le persona (mauvais exemples, à ne jamais produire : "couleur vert", "docteur \
pilule vert langue pourquoi").
- "output" : la réponse que donnerait réellement l'interlocuteur·rice précisé·e \
ci-dessus (pas un·e assistant·e générique), en français correctement écrit, phrase(s) \
complète(s), simple(s) et adaptée(s) à l'âge et au profil du persona (vocabulaire \
et longueur de phrase ajustés, par ex. plus simples pour un enfant que pour un \
adulte), dans le registre FALC, et en respectant les éventuelles contraintes \
spécifiques au thème listées ci-dessus.

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
        theme_interlocuteur=theme.interlocuteur,
        sous_theme_id=sous_theme.id,
        sous_theme_description=sous_theme.description,
        contraintes_bloc=contraintes_bloc,
        persona_age=persona.age,
        persona_profil=persona.profil,
        persona_niveau_langage=persona.niveau_langage,
        persona_moyen_caa=persona.moyen_caa,
        persona_exemple_instruction=persona.exemple_instruction,
    )
