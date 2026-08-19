"""Gabarits de prompt FALC pour la génération d'exemples instruct.

La génération se fait en deux appels séparés par exemple : un appel produit
uniquement l'"instruction" (cf. build_instruction_prompt), un second produit
uniquement l'"output" à partir de l'instruction retenue (cf. build_output_prompt).
Séparer les deux évite qu'un appel unique ne lisse le bruit caractéristique propre au
persona dans l'instruction.
"""

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
tiers connaît. Par défaut, il propose directement 2 ou 3 formulations candidates ; il \
ne pose une question de clarification que si la situation est réellement ambiguë.
- Rôle B : l'assistant répond directement à une vraie question de connaissance \
générale, simple et autosuffisante (tout le contexte nécessaire est déjà dans \
l'instruction). La réponse doit apporter un contenu factuel réel, jamais une esquive \
du type "je vais t'expliquer" sans rien expliquer.

Registre : {registre}

Règles de style à respecter strictement :
{regles}"""


PERSONA_BLOC_TEMPLATE = """Persona :
- Âge : {persona_age}
- Profil : {persona_profil}
- Niveau de langage : {persona_niveau_langage}
- Moyen de CAA utilisé : {persona_moyen_caa}
- Longueur maximale de l'instruction : {persona_longueur_max_mots} mots
- Grammaire attendue : {persona_grammaire}
- Registre lexical : {persona_registre_lexical}
- Bruit caractéristique de ce moyen de CAA : {persona_bruit_caracteristique}
- Registre d'adresse de l'assistant envers ce persona : {persona_registre_adresse}
- Exemple de niveau de langage pour ce persona (à adapter au sous-thème et à \
l'intention ci-dessous, ne pas recopier tel quel) : "{persona_exemple_instruction}\""""


INSTRUCTION_PROMPT_TEMPLATE = """Génère uniquement le champ "instruction" pour cet exemple.

Thème : {theme_label}
Contexte : {theme_contexte}
Sous-thème : {sous_theme_id} — {sous_theme_description}
Angle précis à adopter pour cette instruction : {intention}

{persona_bloc}

{role_instruction_bloc}

Règles impératives pour l'"instruction" :
- Doit rester une phrase à peu près correcte et reconnaissable comme une phrase, même \
simplifiée (mots manquants ou conjugaison approximative tolérés selon le profil du \
persona) — jamais une simple liste de mots-clés juxtaposés sans lien grammatical \
(mauvais exemples, à ne jamais produire : "couleur vert", "docteur pilule vert langue \
pourquoi").
- Auto-suffisance stricte : ne renvoie jamais à un référent que l'assistant n'a pas \
reçu ("cet exercice", "la leçon", "ce document"...). Privilégie les questions \
explicatives (pourquoi, comment, c'est quoi, est-ce que c'est normal, qu'est-ce qui se \
passe si, combien de temps).
- N'invente aucun détail contextuel précis (date, heure, lieu, nom propre, numéro).
{deja_retenues_bloc}
Réponds uniquement avec l'objet JSON demandé : {{"instruction": "..."}}"""


OUTPUT_PROMPT_TEMPLATE = """Génère uniquement le champ "output" pour cet exemple, en réponse à \
l'"instruction" suivante déjà retenue.

Thème : {theme_label}
Contexte : {theme_contexte}
Sous-thème : {sous_theme_id} — {sous_theme_description}
{contraintes_bloc}
{persona_bloc}

Instruction du persona (déjà fixée, ne pas la modifier ni la recopier telle quelle en \
préambule de l'output) : "{instruction}"

{role_output_bloc}

Règles impératives pour l'"output" :
- Français correctement écrit, phrase(s) complète(s), simple(s), registre FALC, \
adapté à l'âge et au profil du persona.
- Ne recopie jamais l'instruction en préambule.
- N'utilise jamais l'impératif pour renvoyer la tâche au persona (interdit par ex. \
"Demande-lui toi-même", "Explique-lui ce que tu ressens") : l'assistant agit lui-même \
en formulant une proposition, jamais en délégant.
- Respecte strictement le registre d'adresse {persona_registre_adresse} envers le \
persona, y compris dans les formulations candidates proposées.
- N'invente aucun détail contextuel précis (date, heure, lieu, nom propre, numéro) et \
aucun contenu que l'assistant n'a pas reçu dans l'instruction.
- Respecte les éventuelles contraintes spécifiques au thème listées ci-dessus.

Réponds uniquement avec l'objet JSON demandé : {{"output": "..."}}"""


ROLE_A_INSTRUCTION_BLOC = """Rôle de l'assistant pour cet exemple : RÔLE A — aider à communiquer avec {theme_interlocuteur}.
Le persona vit une situation réelle (un cours, une tâche, un rendez-vous...) dont \
l'assistant ne connaît PAS le contenu précis. L'"instruction" est le message du \
persona à propos de cette situation, qui doit rester générale (ex. "je ne comprends \
pas la leçon") sans détail que l'assistant serait censé connaître."""

ROLE_B_INSTRUCTION_BLOC = """Rôle de l'assistant pour cet exemple : RÔLE B — répondre à une question de connaissance générale.
L'"instruction" est une vraie question de connaissance générale, simple et \
autosuffisante : tout ce qu'il faut pour y répondre est déjà dans la question, sans \
référence à une situation ou un contenu externe non fourni (ex. "C'est quoi un \
verbe ?", "Combien font 5 et 3 ?", "C'est quand l'automne ?", "Pourquoi le ciel est \
bleu ?")."""


ROLE_A_OUTPUT_BLOC_PROPOSITION = """Rôle de l'assistant pour cet exemple : RÔLE A — aider à communiquer avec {theme_interlocuteur}.
Comportement attendu (cas par défaut, situation pas ambiguë) : propose directement 2 \
ou 3 formulations candidates que le persona pourrait dire ou écrire à \
{theme_interlocuteur}. Ne pose PAS de question de clarification ici. Jamais une \
réponse au contenu que l'assistant ne connaît pas.
Exemple correct : "Tu pourrais dire : « Je n'ai pas compris, tu peux réexpliquer ? » \
ou bien « Est-ce que tu peux recommencer plus lentement ? »"
Exemple interdit (hallucination de contenu) : "Bien sûr, je vais te réexpliquer, tu \
veux que je commence par la fin ou le début ?\""""

ROLE_A_OUTPUT_BLOC_CLARIFICATION = """Rôle de l'assistant pour cet exemple : RÔLE A — aider à communiquer avec {theme_interlocuteur}.
Comportement attendu (cas minoritaire, situation réellement ambiguë) : pose une \
question de clarification courte pour préciser ce que le persona veut dire à \
{theme_interlocuteur}, avant de proposer une formulation. Jamais une réponse au \
contenu que l'assistant ne connaît pas.
Exemple correct : "Tu veux dire à {theme_interlocuteur} que tu n'as pas compris, ou \
que tu es fatigué ?"
Exemple interdit (hallucination de contenu) : "Bien sûr, je vais te réexpliquer, tu \
veux que je commence par la fin ou le début ?\""""

ROLE_B_OUTPUT_BLOC = """Rôle de l'assistant pour cet exemple : RÔLE B — répondre à une question de connaissance générale.
Donne une vraie réponse factuelle, courte, juste et adaptée à l'âge et au profil du \
persona — jamais une esquive du type "je vais t'expliquer" sans contenu réel."""


def build_system_prompt(style_guide: StyleGuide) -> str:
    regles = "\n".join(f"- {regle}" for regle in style_guide.regles)
    return SYSTEM_PROMPT.format(registre=style_guide.registre, regles=regles)


def _persona_bloc(persona: Persona, exemple_instruction: str) -> str:
    return PERSONA_BLOC_TEMPLATE.format(
        persona_age=persona.age,
        persona_profil=persona.profil,
        persona_niveau_langage=persona.niveau_langage,
        persona_moyen_caa=persona.moyen_caa,
        persona_longueur_max_mots=persona.longueur_max_mots,
        persona_grammaire=persona.grammaire,
        persona_registre_lexical=persona.registre_lexical,
        persona_bruit_caracteristique=persona.bruit_caracteristique,
        persona_registre_adresse=persona.registre_adresse,
        persona_exemple_instruction=exemple_instruction,
    )


def build_instruction_prompt(
    theme: Theme,
    sous_theme: SousTheme,
    persona: Persona,
    role: RoleType,
    intention: str,
    exemple_instruction: str,
    instructions_deja_retenues: list[str],
) -> str:
    if role == "A":
        role_bloc = ROLE_A_INSTRUCTION_BLOC.format(
            theme_interlocuteur=theme.interlocuteur_pour(persona.id)
        )
    else:
        role_bloc = ROLE_B_INSTRUCTION_BLOC

    deja_retenues_bloc = ""
    if instructions_deja_retenues:
        lignes = "\n".join(f'- "{i}"' for i in instructions_deja_retenues)
        deja_retenues_bloc = (
            "- Instructions déjà retenues pour cette même combinaison thème / "
            "sous-thème / persona : ne produis surtout pas une instruction qui leur "
            f"ressemble (même formulation, même angle) :\n{lignes}\n"
        )

    return INSTRUCTION_PROMPT_TEMPLATE.format(
        theme_label=theme.label,
        theme_contexte=theme.contexte,
        sous_theme_id=sous_theme.id,
        sous_theme_description=sous_theme.description,
        intention=intention,
        persona_bloc=_persona_bloc(persona, exemple_instruction),
        role_instruction_bloc=role_bloc,
        deja_retenues_bloc=deja_retenues_bloc,
    )


def build_output_prompt(
    theme: Theme,
    sous_theme: SousTheme,
    persona: Persona,
    role: RoleType,
    instruction: str,
    exemple_instruction: str,
    demande_clarification: bool = False,
) -> str:
    contraintes_bloc = ""
    if theme.contraintes_specifiques:
        lignes = "\n".join(f"- {c}" for c in theme.contraintes_specifiques)
        contraintes_bloc = f"Contraintes spécifiques au thème :\n{lignes}\n"

    if role == "A":
        theme_interlocuteur = theme.interlocuteur_pour(persona.id)
        bloc_template = (
            ROLE_A_OUTPUT_BLOC_CLARIFICATION if demande_clarification else ROLE_A_OUTPUT_BLOC_PROPOSITION
        )
        role_bloc = bloc_template.format(theme_interlocuteur=theme_interlocuteur)
    else:
        role_bloc = ROLE_B_OUTPUT_BLOC

    return OUTPUT_PROMPT_TEMPLATE.format(
        theme_label=theme.label,
        theme_contexte=theme.contexte,
        sous_theme_id=sous_theme.id,
        sous_theme_description=sous_theme.description,
        contraintes_bloc=contraintes_bloc,
        persona_bloc=_persona_bloc(persona, exemple_instruction),
        instruction=instruction,
        role_output_bloc=role_bloc,
        persona_registre_adresse=persona.registre_adresse,
    )
