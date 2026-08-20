"""Gabarits de prompt FALC pour la génération d'exemples instruct.

La génération se fait en deux appels séparés par exemple : un appel produit
uniquement l'"instruction" (cf. build_instruction_prompt), un second produit
uniquement l'"output" à partir de l'instruction retenue (cf. build_output_prompt).
Séparer les deux évite qu'un appel unique ne lisse le bruit caractéristique propre au
persona dans l'instruction.

L'assistant a un seul comportement, décliné en deux cas (cf. CasType) : RÉPONSE
(répondre directement, de façon factuelle) ou RENVOI (dire qu'il ne peut pas répondre
et indiquer vers qui se tourner). Le critère de bascule n'est jamais le sujet de la
question mais la source de la réponse : dépend-elle de cette personne, de ce lieu, de
ce moment ? Si oui, RENVOI ; sinon, RÉPONSE.
"""

from __future__ import annotations

import random
from typing import Literal

from panta_generate_data_instruct.schemas import Persona, SousTheme, StyleGuide, Theme

CasType = Literal["reponse", "renvoi"]

SYSTEM_PROMPT = """Tu es un générateur de données d'entraînement pour un assistant de \
Communication Alternative et Améliorée (CAA), utilisé par des personnes en situation \
de handicap de la communication.

Chaque exemple simule un échange réel : "instruction" est le message composé par le \
persona via son moyen de CAA, et "output" est la réponse de l'ASSISTANT CAA lui-même \
(pas celle d'un tiers réel comme un·e médecin, un·e enseignant·e ou un·e proche). \
L'assistant est un outil de communication que le persona utilise : il ne participe \
jamais physiquement à la situation vécue par le persona (il n'est pas en classe, pas \
au rendez-vous médical, etc.) et ne connaît donc JAMAIS un contenu qui ne lui a pas \
été donné dans l'instruction (le contenu d'un cours, ce qu'un tiers a dit, un fait \
précis non fourni...). Halluciner un tel contenu est une erreur grave à éviter \
absolument.

L'assistant a un seul comportement, décliné en deux cas, précisé dans le prompt \
utilisateur :
- Cas RÉPONSE : l'assistant répond directement à la question, de façon factuelle, \
simple et courte.
- Cas RENVOI : l'assistant n'est pas en position de savoir. Il dit qu'il ne peut pas \
répondre, et indique vers qui se tourner (le médecin, l'enseignant·e, la personne en \
face de lui, un proche, selon la situation).

Le critère de bascule entre les deux cas n'est JAMAIS le sujet de la question (santé, \
école, famille...), mais la source de la réponse :
Question à se poser : la réponse dépend-elle de cette personne, de ce lieu, ou de ce \
moment précis ? Si oui → RENVOI. Sinon → RÉPONSE.
Exemples : "C'est quoi une IRM ?" → RÉPONSE (connaissance générale). "Pourquoi ma \
langue est verte depuis le médicament ?" → RENVOI vers le médecin (dépend de cette \
personne). "Combien de temps dure un rendez-vous médical ?" → RÉPONSE (fait général). \
"Où sont les toilettes ?" → RENVOI vers une personne présente (dépend de ce lieu). \
"Combien de temps de retard ?" → RENVOI vers l'agent (dépend de ce moment). Cette \
logique s'applique à tous les thèmes, pas seulement au médical.

Dans le cas RENVOI, l'assistant ne dit jamais si c'est normal, grave ou bénin, et ne \
propose jamais de cause ni d'explication partielle : il dit seulement qu'il ne peut \
pas répondre, et vers qui se tourner. Rien d'autre.

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

{cas_instruction_bloc}

Règles impératives pour l'"instruction" :
- Doit rester une phrase à peu près correcte et reconnaissable comme une phrase, même \
simplifiée (mots manquants ou conjugaison approximative tolérés selon le profil du \
persona) — jamais une simple liste de mots-clés juxtaposés sans lien grammatical \
(mauvais exemples, à ne jamais produire : "couleur vert", "docteur pilule vert langue \
pourquoi") et jamais une combinaison de mots qui n'a pas de sens (mauvais exemples, à \
ne jamais produire : "Combien de temps pour enfiler la ville ?", "je trouve comment \
enterrer bien usage télévision") : chaque mot choisi doit avoir un rapport clair et \
compréhensible avec le sous-thème et l'angle demandés.
- Auto-suffisance stricte : ne renvoie jamais à un référent que l'assistant n'a pas \
reçu ("cet exercice", "la leçon", "ce document"...). Tout démonstratif (ce, cet, \
cette, ces) doit obligatoirement pointer vers un mot déjà présent plus tôt dans \
l'instruction elle-même ; sinon utilise un article indéfini ("un examen", "une \
tâche") plutôt qu'un démonstratif. Privilégie les questions explicatives (pourquoi, \
comment, c'est quoi, est-ce que c'est normal, qu'est-ce qui se passe si, combien de \
temps).
- N'invente aucun détail contextuel précis (date, heure, lieu, nom propre, numéro).
- Respecte strictement le champ "Grammaire attendue" du persona ci-dessus : un \
persona avec une grammaire correcte (conjugaison correcte, phrases complètes) ne \
produit jamais un enchaînement de mots sans verbe conjugué ni lien grammatical, même \
si l'instruction reste courte.
- Reprends l'exemple de niveau de langage du persona uniquement comme référence de \
style (longueur, bruit, grammaire) : ne le recopie jamais tel quel ni presque tel \
quel, produis un contenu différent adapté au sous-thème et à l'angle demandés.
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

{cas_output_bloc}

Règles impératives pour l'"output" :
- Français impeccable : orthographe et conjugaison correctes de chaque verbe, \
phrase(s) complète(s), simple(s), registre FALC, adapté à l'âge et au profil du \
persona. Le bruit (mots manquants, syntaxe télégraphique...) appartient uniquement à \
l'instruction du persona ; il ne doit jamais apparaître dans l'output, même quand \
l'instruction est bruitée ou télégraphique.
- Ne recopie jamais l'instruction en préambule.
- N'utilise jamais l'impératif pour renvoyer la tâche au persona (interdit par ex. \
"Demande-lui toi-même", "Il faut demander à ton camarade") : l'assistant ne délègue \
jamais sa réponse au persona.
- Ne s'engage jamais physiquement à la place du tiers réel avec qui le persona \
communique (interdit par ex. "je viens vous aider à porter ça", "j'arrive tout de \
suite") : l'assistant n'agit jamais lui-même dans la situation vécue par le persona.
- Respecte strictement le registre d'adresse {persona_registre_adresse} envers le \
persona.
- N'invente aucun détail contextuel précis (date, heure, lieu, nom propre, numéro) et \
aucun contenu que l'assistant n'a pas reçu dans l'instruction. Tout démonstratif (ce, \
cet, cette, ces) doit pointer vers un mot déjà présent dans l'instruction ou plus tôt \
dans l'output ; sinon utilise un article indéfini.
- Termine toujours par une ponctuation finale (. ! ou ?) et referme tout guillemet \
ouvert : ne coupe jamais l'output en cours de phrase ou en plein guillemet.
- Ne fais jamais référence, dans l'output, au critère de bascule lui-même ou au fait \
que la question soit générale ou dépendante du contexte (interdit par ex. "Cela ne \
dépend de personne en particulier, c'est valable pour tout le monde") : réponds ou \
renvoie directement, sans jamais commenter pourquoi tu réponds ou renvoies.
- Respecte les éventuelles contraintes spécifiques au thème listées ci-dessus.

Réponds uniquement avec l'objet JSON demandé : {{"output": "..."}}"""


CAS_REPONSE_INSTRUCTION_BLOC = """Cas de cet exemple : RÉPONSE.
L'"instruction" doit être une vraie question dont la réponse est exactement la même \
pour n'importe qui, n'importe où, n'importe quand (ex. "C'est quoi un verbe ?", \
"Combien font 5 et 3 ?", "Combien de temps dure un rendez-vous médical en général ?"). \
Si en écrivant l'instruction tu te rends compte que la réponse dépendrait en réalité \
de cette personne, de ce lieu ou de ce moment précis, ce n'est PAS ce cas : reformule \
l'angle pour rester sur une vraie question de portée générale."""

CAS_RENVOI_INSTRUCTION_BLOC = """Cas de cet exemple : RENVOI.
L'"instruction" porte sur quelque chose dont la réponse dépend de cette personne, de \
ce lieu ou de ce moment (ex. un symptôme ressenti, où se trouve un objet ici, la durée \
d'un retard maintenant, pourquoi un proche réagit d'une certaine façon aujourd'hui). \
L'assistant qui recevrait cette instruction ne peut pas connaître la réponse à \
l'avance, quel que soit son savoir général.
La spécificité qui rend la réponse dépendante de la personne, du lieu ou du moment \
doit toujours venir d'un indexical (maintenant, ici, mon, ma, encore, aujourd'hui...), \
jamais d'un détail inventé. Correct : "Est-ce que c'est bientôt mon tour ?", "Combien \
de temps ça va encore durer ?". Interdit (détail inventé) : "Comment Marie a résolu \
son problème ?", "Que veut dire le mot dissolution dans l'explication ?"."""


CAS_REPONSE_OUTPUT_BLOC = """Cas de cet exemple : RÉPONSE.
Donne une vraie réponse factuelle, courte, juste et adaptée à l'âge et au profil du \
persona — jamais une esquive du type "je vais t'expliquer" sans contenu réel. Cette \
réponse doit être valable pour n'importe qui, n'importe où, n'importe quand : si tu \
constates qu'elle dépend en réalité du lieu, du moment ou de la personne, ne réponds \
jamais comme si tu connaissais ce contexte précis (interdit par ex. "les toilettes \
sont sur la gauche", "c'est normal que la machine fasse ce bruit").
Consigne de forme pour cette réponse précise (varie d'un exemple à l'autre, à \
respecter ici) : {format_directive}"""

CAS_RENVOI_OUTPUT_BLOC = """Cas de cet exemple : RENVOI — l'assistant n'est pas en position de savoir.
Structure attendue, rien d'autre : une phrase qui nomme l'objet précis de la question \
(pas la question en général), suivie d'une phrase qui indique vers qui se tourner.
La première phrase doit nommer l'objet précis de la question. Interdit (trop \
générique) : "Je ne peux pas répondre à cela", "Je ne peux pas répondre à ça", "Je \
n'ai pas cette information". Attendu : "Je ne sais pas pourquoi ta langue est verte", \
"Je ne sais pas combien de temps le train va être en retard".
Destinataires possibles pour ce thème et ce persona : {destinataires_possibles}. \
Choisis un seul destinataire, celui qui est le plus pertinent pour cette question \
précise (ex. une question à propos d'un camarade se renvoie vers ce camarade, pas \
systématiquement vers l'enseignant·e) ; n'énumère jamais plusieurs destinataires et ne \
reprends jamais de parenthèses.
Aucune formulation candidate, aucune explication partielle, aucun avis sur \
normal/grave/bénin, aucune cause proposée, même partielle.
Modèle de style ci-dessous pour le ton et la deuxième phrase uniquement (le \
destinataire y est encore générique : remplace-le par le destinataire choisi, et \
remplace toujours l'ouverture générique par une phrase qui nomme l'objet précis de \
CETTE question, jamais mot pour mot) : "{variante_renvoi}\""""


# Variantes de la phrase "je ne peux pas répondre à ça", tirées au sort pour éviter
# qu'un patron unique ("Tu veux que je t'aide ?"-like) ne se répète sur tout le
# sous-ensemble RENVOI. {interlocuteur} est rempli avec Theme.interlocuteurs_pour().
RENVOI_PHRASES_VARIANTES = [
    "Ça, je ne peux pas te le dire. C'est {interlocuteur} qui peut te répondre.",
    "Je ne suis pas en mesure de répondre à ça. Il vaut mieux demander à {interlocuteur}.",
    "Ce n'est pas à moi de répondre à cette question. Pose-la à {interlocuteur}.",
    "Je n'ai pas cette information. C'est une question à poser à {interlocuteur}.",
    "Là, je ne peux pas t'aider directement. {interlocuteur_maj} pourra te répondre.",
    "Ce n'est pas quelque chose que je peux savoir. Demande plutôt à {interlocuteur}.",
    "Je ne peux pas te donner de réponse sur ce point. Il n'y a que {interlocuteur} qui puisse te répondre.",
    "Ça dépend de quelque chose que je ne connais pas, donc je ne peux pas répondre. {interlocuteur_maj} saura te dire.",
]


# Variantes de forme pour le cas RÉPONSE, tirées au sort pour éviter que les outputs
# ne commencent tous par le même enrobage (constat observé sur la v2).
FORMATS_REPONSE_VARIANTES = [
    "réponds directement, sans aucune formule d'introduction, en une seule phrase courte.",
    "commence par une très courte phrase d'introduction (2-4 mots) qui annonce que la réponse arrive, sans réutiliser toujours la même formule, puis donne la réponse.",
    "réponds en deux phrases courtes plutôt qu'une seule, sans préambule.",
    "réponds directement en une seule phrase, sans aucun préambule ni formule de politesse.",
    "réponds en une phrase de longueur moyenne, ni très courte ni longue, sans préambule.",
    "donne la réponse directement puis ajoute une très courte phrase complémentaire si utile, sans préambule.",
]


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
    cas: CasType,
    intention: str,
    exemple_instruction: str,
    instructions_deja_retenues: list[str],
) -> str:
    cas_bloc = CAS_REPONSE_INSTRUCTION_BLOC if cas == "reponse" else CAS_RENVOI_INSTRUCTION_BLOC

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
        cas_instruction_bloc=cas_bloc,
        deja_retenues_bloc=deja_retenues_bloc,
    )


def choisir_variante_renvoi(theme: Theme, persona: Persona) -> str:
    """Tire au sort une variante de phrase de renvoi (secours stylistique) et la
    remplit avec un destinataire représentatif du thème pour ce persona ; le modèle
    doit ensuite choisir lui-même le destinataire le plus pertinent pour la question
    précise (cf. CAS_RENVOI_OUTPUT_BLOC), celui-ci n'est qu'un exemple de ton."""
    interlocuteur = random.choice(theme.interlocuteurs_pour(persona.id))
    variante = random.choice(RENVOI_PHRASES_VARIANTES)
    return variante.format(
        interlocuteur=interlocuteur,
        interlocuteur_maj=interlocuteur[0].upper() + interlocuteur[1:],
    )


def _destinataires_possibles(theme: Theme, persona: Persona) -> str:
    return ", ".join(theme.interlocuteurs_pour(persona.id))


def choisir_format_reponse() -> str:
    """Tire au sort une consigne de forme pour un output du cas RÉPONSE."""
    return random.choice(FORMATS_REPONSE_VARIANTES)


def build_output_prompt(
    theme: Theme,
    sous_theme: SousTheme,
    persona: Persona,
    cas: CasType,
    instruction: str,
    exemple_instruction: str,
    variante_renvoi: str | None = None,
    format_directive: str | None = None,
) -> str:
    contraintes_bloc = ""
    if theme.contraintes_specifiques:
        lignes = "\n".join(f"- {c}" for c in theme.contraintes_specifiques)
        contraintes_bloc = f"Contraintes spécifiques au thème :\n{lignes}\n"

    if cas == "renvoi":
        cas_bloc = CAS_RENVOI_OUTPUT_BLOC.format(
            destinataires_possibles=_destinataires_possibles(theme, persona),
            variante_renvoi=variante_renvoi or choisir_variante_renvoi(theme, persona),
        )
    else:
        cas_bloc = CAS_REPONSE_OUTPUT_BLOC.format(
            format_directive=format_directive or choisir_format_reponse()
        )

    return OUTPUT_PROMPT_TEMPLATE.format(
        theme_label=theme.label,
        theme_contexte=theme.contexte,
        sous_theme_id=sous_theme.id,
        sous_theme_description=sous_theme.description,
        contraintes_bloc=contraintes_bloc,
        persona_bloc=_persona_bloc(persona, exemple_instruction),
        instruction=instruction,
        cas_output_bloc=cas_bloc,
        persona_registre_adresse=persona.registre_adresse,
    )


# Appel de validation binaire de cohérence sémantique, entre la génération de
# l'instruction et celle de l'output (cf. generate._filtrer_coherence_semantique).
# Contrairement au filtre par similarité d'embeddings (qui détecte une dérive
# thématique mais pas un charabia thématiquement proche), cet appel cible le non-sens
# véritable (ex. "Combien de temps pour enfiler la ville ?").
COHERENCE_SYSTEM_PROMPT = """Tu vérifies si une phrase en français a un sens \
compréhensible, en tant que message composé par une personne via un système de \
Communication Alternative et Améliorée (CAA). Une phrase simplifiée, télégraphique ou \
avec des mots manquants PEUT avoir du sens (ex. "Ventre fait mal, pourquoi ?" a du \
sens, malgré la grammaire simplifiée). N'A PAS de sens une combinaison de mots qui ne \
correspond à aucune situation compréhensible, même si chaque mot pris isolément \
existe (ex. "Combien de temps pour enfiler la ville ?", "je trouve comment enterrer \
bien usage télévision")."""

COHERENCE_CHECK_PROMPT_TEMPLATE = """Cette phrase a-t-elle un sens compréhensible en français ?

Phrase : "{instruction}"

Réponds uniquement avec l'objet JSON demandé : {{"sens": true ou false}}"""


def build_coherence_check_prompt(instruction: str) -> str:
    return COHERENCE_CHECK_PROMPT_TEMPLATE.format(instruction=instruction)
