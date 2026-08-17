"""Gabarits de prompt FALC pour la génération d'exemples instruct."""

from __future__ import annotations

from typing import Literal

from panta_generate_data_instruct.schemas import Persona, SousTheme, StyleGuide, Theme

RoleType = Literal["A", "B"]

SYSTEM_PROMPT = """Tu es un générateur de données d'entraînement pour un assistant de \
Communication Alternative et Améliorée (CAA), utilisé par des personnes en situation \
de handicap de la communication.

Chaque exemple simule un échange réel : "instruction" est le message composé par le \
persona via son moyen de CAA, et "output" est la réponse de l'ASSISTANT CAA lui-même \
(pas celle d'un tiers réel comme un·e médecin, un·e enseignant·e ou un·e proche). \
L'assistant est un outil de communication que le persona utilise pour interagir avec \
des tiers réels : il ne participe jamais physiquement à la situation vécue par le \
persona (il n'est pas en classe, pas au rendez-vous médical, etc.) et ne connaît donc \
JAMAIS un contenu qui ne lui a pas été donné dans l'instruction (le contenu d'un \
cours, ce qu'un tiers a dit, un fait précis non fourni...). Halluciner un tel contenu \
est une erreur grave à éviter absolument.

Chaque exemple relève de l'un de ces deux rôles, précisé dans le prompt utilisateur :
- Rôle A : l'assistant aide le persona à formuler, reformuler ou clarifier un message \
à adresser à un tiers réel (professionnel·le, proche, enseignant·e, camarade...). Il \
ne répond jamais à la place de ce tiers et n'invente jamais le contenu que seul ce \
tiers connaît.
- Rôle B : l'assistant répond directement à une vraie question de connaissance \
générale, simple et autosuffisante (tout le contexte nécessaire est déjà dans \
l'instruction). La réponse doit apporter un contenu factuel réel, jamais une esquive \
du type "je vais t'expliquer" sans rien expliquer.

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
- Longueur maximale de l'instruction : {persona_longueur_max_mots} mots
- Grammaire attendue : {persona_grammaire}
- Registre lexical : {persona_registre_lexical}
- Bruit caractéristique de ce moyen de CAA : {persona_bruit_caracteristique}
- Exemple de niveau de langage pour ce persona (à adapter au sous-thème, ne pas \
recopier tel quel) : "{persona_exemple_instruction}"

{role_bloc}

Réponds uniquement avec l'objet JSON demandé."""


ROLE_A_BLOC = """Rôle de l'assistant pour cet exemple : RÔLE A — aider à communiquer avec {theme_interlocuteur}.
Le persona vit une situation réelle (un cours, une tâche, un rendez-vous...) dont \
l'assistant ne connaît PAS le contenu précis. L'assistant ne doit donc jamais \
halluciner ni inventer ce contenu, ni répondre à la place de {theme_interlocuteur}.
- "instruction" : le message du persona à propos de cette situation, qui peut rester \
général (ex. "je ne comprends pas la leçon") sans détail que l'assistant serait censé \
connaître. Respecte la longueur, la grammaire et le registre lexical du persona \
donnés ci-dessus.
- "output" : l'assistant aide à formuler, reformuler ou clarifier ce que le persona \
veut dire à {theme_interlocuteur} — jamais une réponse au contenu que l'assistant ne \
connaît pas. Exemple correct : "Tu veux que je t'aide à dire à la maîtresse que tu \
n'as pas compris ?" Exemple interdit (hallucination de contenu) : "Bien sûr, je vais \
te réexpliquer, tu veux que je commence par la fin ou le début ?\""""

ROLE_B_BLOC = """Rôle de l'assistant pour cet exemple : RÔLE B — répondre à une question de connaissance générale.
- "instruction" : une vraie question de connaissance générale, simple et \
autosuffisante : tout ce qu'il faut pour y répondre est déjà dans la question, sans \
référence à une situation ou un contenu externe non fourni (ex. "C'est quoi un \
verbe ?", "Combien font 5 et 3 ?", "C'est quand l'automne ?", "Pourquoi le ciel est \
bleu ?"). Respecte la longueur, la grammaire et le registre lexical du persona donnés \
ci-dessus.
- "output" : une vraie réponse factuelle, courte, juste et adaptée à l'âge et au \
profil du persona — jamais une esquive du type "je vais t'expliquer" sans contenu \
réel."""


GENERAL_INSTRUCTION_RULES = """Génère un exemple d'échange pour ce persona et ce sous-thème, dans le rôle précisé \
ci-dessus. Dans tous les cas :
- L'"instruction" doit toujours rester une phrase à peu près correcte et \
reconnaissable comme une phrase, même simplifiée (mots manquants ou conjugaison \
approximative tolérés selon le profil du persona) — jamais une simple liste de \
mots-clés juxtaposés sans lien grammatical (mauvais exemples, à ne jamais produire : \
"couleur vert", "docteur pilule vert langue pourquoi").
- L'"output" est en français correctement écrit, phrase(s) complète(s), simple(s) et \
adaptée(s) à l'âge et au profil du persona, dans le registre FALC, et respecte les \
éventuelles contraintes spécifiques au thème listées ci-dessus."""


def build_system_prompt(style_guide: StyleGuide) -> str:
    regles = "\n".join(f"- {regle}" for regle in style_guide.regles)
    return SYSTEM_PROMPT.format(registre=style_guide.registre, regles=regles)


def build_role_bloc(role: RoleType, theme: Theme) -> str:
    if role == "A":
        bloc = ROLE_A_BLOC.format(theme_interlocuteur=theme.interlocuteur)
    else:
        bloc = ROLE_B_BLOC
    return f"{bloc}\n\n{GENERAL_INSTRUCTION_RULES}"


def build_user_prompt(theme: Theme, sous_theme: SousTheme, persona: Persona, role: RoleType) -> str:
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
        persona_longueur_max_mots=persona.longueur_max_mots,
        persona_grammaire=persona.grammaire,
        persona_registre_lexical=persona.registre_lexical,
        persona_bruit_caracteristique=persona.bruit_caracteristique,
        persona_exemple_instruction=persona.exemple_instruction,
        role_bloc=build_role_bloc(role, theme),
    )
