"""Filtres de validation post-génération.

Complètent les consignes de prompt.py par des contrôles code qui REJETTENT
effectivement un candidat (au lieu de se contenter de le signaler) : les consignes
seules dans le prompt se sont montrées insuffisantes lors de l'audit du dernier
échantillon (démonstratifs non introduits, réassurances médicales interdites,
engagement physique de l'assistant, délégation à l'impératif, frontière rôle A/B,
quelques fautes de français récurrentes).

Ces filtres sont des heuristiques regex volontairement simples : ils ne remplacent pas
une relecture humaine ni un correcteur grammatical, mais ciblent des patrons concrets
observés dans l'audit.
"""

from __future__ import annotations

import re

_DEMONSTRATIF_RE = re.compile(r"\b(ce|cet|cette|ces)\s+([a-zàâäéèêëïîôöùûüç-]+)", re.IGNORECASE)

# Mots après lesquels "ce/cette/..." n'est pas un déterminant démonstratif introduisant
# un nom (pronom relatif, verbe...), pour limiter les faux positifs. "qu" couvre "ce
# qu'il/qu'elle..." (l'apostrophe coupe le mot avant la voyelle).
_STOPWORDS_DEMONSTRATIF = {"que", "qu", "qui", "dont", "sont", "est", "sera", "soit"}

# Locution figée avec démonstratif qui ne renvoie à aucun référent externe (donc
# jamais une violation de l'auto-suffisance), à exempter explicitement.
_LOCUTION_CE_MOMENT_RE = re.compile(r"\b(en|à)\s+ce\s+moment\b", re.IGNORECASE)


def demonstratifs_non_introduits(texte: str) -> list[str]:
    """Renvoie les groupes "démonstratif + nom" dont le nom n'apparaît nulle part plus
    tôt dans le même texte. Un démonstratif (ce/cet/cette/ces) doit toujours pointer
    vers un référent déjà introduit dans le texte lui-même ; sinon il renvoie à un
    contenu externe que l'assistant n'a jamais reçu (cf. règle d'auto-suffisance)."""
    if _LOCUTION_CE_MOMENT_RE.search(texte):
        texte = _LOCUTION_CE_MOMENT_RE.sub("", texte)

    trouves: list[str] = []
    for m in _DEMONSTRATIF_RE.finditer(texte):
        determinant, nom = m.groups()
        nom_lower = nom.lower()
        if nom_lower in _STOPWORDS_DEMONSTRATIF:
            continue
        avant = texte[: m.start()]
        if not re.search(rf"\b{re.escape(nom_lower)}\b", avant, re.IGNORECASE):
            trouves.append(f"{determinant} {nom}")
    return trouves


_REASSURANCE_SYMPTOME_RE = re.compile(
    r"\bc'est (souvent |sûrement |probablement |généralement )?(normal|grave|bénin|fréquent|rien du tout|rien)\b"
    r"|\b(cela|ça) (ne )?(veut|peut) (pas )?(dire|être)\b"
    r"|\b(c'est|ça peut être|cela peut être) (à cause|lié) (de|à|du|des)\b"
    r"|\bpas (la peine|grave) de (t'|vous )?inquiéter\b",
    re.IGNORECASE,
)


def reponse_symptome_interdite(texte: str) -> bool:
    """Détecte une réassurance ou une explication de cause à propos d'un symptôme :
    toujours interdit pour l'assistant, qui ne dit jamais si c'est normal/grave/bénin
    et n'en propose jamais la cause (règle globale symptômes, cf. SYSTEM_PROMPT)."""
    return bool(_REASSURANCE_SYMPTOME_RE.search(texte))


_ENGAGEMENT_PHYSIQUE_RE = re.compile(
    r"\bje (viens|vais|arrive|passe)\b[^.!?]{0,40}\b(aider|porter|chercher|apporter|venir)\b"
    r"|\bj'arrive\b"
    r"|\bje suis (là|en route)\b",
    re.IGNORECASE,
)


def engagement_physique_assistant(texte: str) -> bool:
    """Détecte l'assistant s'engageant lui-même dans une action physique à la place
    d'un tiers réel : interdit, l'assistant ne participe jamais physiquement à la
    situation vécue par le persona (il n'est pas au rendez-vous, pas en classe...)."""
    return bool(_ENGAGEMENT_PHYSIQUE_RE.search(texte))


_IMPERATIF_DELEGATION_RE = re.compile(
    r"(^|[.!?]\s+)(demande|explique|dis|va|parle|préviens|signale|raconte)[- ](lui|leur|toi)\b"
    r"|\bil (te |vous |lui |leur )?faut\b[^.!?]{0,20}\b(demander|expliquer|dire|parler|signaler)\b",
    re.IGNORECASE,
)


def imperatif_delegation(texte: str) -> bool:
    """Détecte un impératif (ou tournure équivalente, ex. "il faut demander...")
    renvoyant la tâche au persona (ex. "Demande-lui toi-même", "Il faut d'abord
    demander à ton camarade") : interdit, l'assistant agit lui-même en formulant une
    proposition, jamais en délégant la tâche au persona."""
    return bool(_IMPERATIF_DELEGATION_RE.search(texte))


_COMMENT_DIRE_RE = re.compile(r"\bcomment\s+(dire|demander|expliquer|formuler)\b", re.IGNORECASE)


def instruction_devrait_etre_role_a(instruction: str) -> bool:
    """Détecte une instruction de la forme "comment dire/demander/expliquer..." qui
    relève toujours du rôle A (aide à formuler un message), jamais du rôle B, même si
    le sujet semble général."""
    return bool(_COMMENT_DIRE_RE.search(instruction))


# Élision manquante : me/te/se/le/la/ne + mot commençant par une voyelle ou un h muet
# (ex. "me empêche" au lieu de "m'empêche").
_ELISION_MANQUANTE_RE = re.compile(
    r"\b(me|te|se|le|la|ne|je)\s+([aeiouyhéèêàâîïôûAEIOUYHÉÈÊÀÂÎÏÔÛ]\w*)", re.IGNORECASE
)

# Modal (veux/peux/dois/veut/peut/doit...) suivi d'un participe passé en -é au lieu de
# l'infinitif en -er (ex. "je veux porté" au lieu de "je veux porter").
_MODAL_PARTICIPE_FAUTIF_RE = re.compile(
    r"\b(veux|veut|peux|peut|dois|doit|voudrais|voudrait|aimerais|aimerait)\s+(\w+é)\b",
    re.IGNORECASE,
)


def incorrections_frequentes(texte: str) -> list[str]:
    """Détecte un sous-ensemble de fautes de français récurrentes observées dans
    l'audit (élision manquante, confusion participe/infinitif après un modal).
    Heuristique partielle : ne couvre pas les mots inventés ni les conjugaisons
    fautives arbitraires, qui nécessiteraient un correcteur grammatical dédié."""
    trouves: list[str] = []
    for m in _ELISION_MANQUANTE_RE.finditer(texte):
        trouves.append(m.group(0))
    for m in _MODAL_PARTICIPE_FAUTIF_RE.finditer(texte):
        mot = m.group(2)
        # "aimé", "gêné"... en fin de proposition après un modal restent souvent
        # fautifs (infinitif attendu), mais on exclut les participes courants employés
        # comme adjectifs pour limiter les faux positifs.
        if mot.lower() not in {"été", "fâché", "fatigué", "inquiété", "gêné"}:
            trouves.append(m.group(0))
    return trouves
