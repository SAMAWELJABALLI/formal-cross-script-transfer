"""Script detection, Arabizi normalisation, a rule-based deromaniser, and Maghrebi
negation rules for building the (evaluation-only) contrast set.

The deromaniser is a deliberately simple baseline for the "canonicalise the input"
condition (the text analogue of TextPro-SLM's input-side principle). It is NOT a
substitute for human transliteration and is never used to create invariance pairs
labelled "human".
"""

from __future__ import annotations

import re
import unicodedata

ARABIC_RANGES = [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
                 (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]


def _is_arabic(ch: str) -> bool:
    o = ord(ch)
    return any(a <= o <= b for a, b in ARABIC_RANGES)


def arabic_script_ratio(text: str) -> float:
    """Share of alphabetic characters that are Arabic-script."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(_is_arabic(c) for c in letters) / len(letters)


def detect_script(text: str, threshold: float = 0.5) -> str:
    r = arabic_script_ratio(text)
    if not any(c.isalpha() for c in text):
        return "none"
    if r >= threshold:
        return "arab"
    if r == 0:
        return "latn"
    return "mixed"


_REPEAT = re.compile(r"(.)\1{2,}")


def normalize_arabizi(text: str) -> str:
    """Lowercase, NFKC, collapse runs of 3+ identical characters to 2, squeeze spaces.
    Digits are kept: in Arabizi 3/7/9/5/2 are letters, so they carry meaning."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _REPEAT.sub(r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


_AR_DIACRITICS = re.compile(r"[ً-ْٰـ]")


def normalize_arabic(text: str) -> str:
    """Remove diacritics and tatweel; unify alef/ya/ta-marbuta variants."""
    t = unicodedata.normalize("NFKC", text)
    t = _AR_DIACRITICS.sub("", t)
    t = re.sub("[إأآا]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه")
    t = _REPEAT.sub(r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


# Longest-match-first mapping, Maghrebi Arabizi conventions.
_DEROM = [
    ("ch", "ش"), ("sh", "ش"), ("gh", "غ"), ("kh", "خ"), ("th", "ث"), ("dh", "ذ"),
    ("ou", "و"), ("oo", "و"), ("ee", "ي"),
    ("3", "ع"), ("7", "ح"), ("9", "ق"), ("5", "خ"), ("2", "ء"), ("8", "غ"),
    ("a", "ا"), ("b", "ب"), ("t", "ت"), ("j", "ج"), ("d", "د"), ("r", "ر"),
    ("z", "ز"), ("s", "س"), ("f", "ف"), ("q", "ق"), ("k", "ك"), ("l", "ل"),
    ("m", "م"), ("n", "ن"), ("h", "ه"), ("w", "و"), ("y", "ي"), ("i", "ي"),
    ("e", ""), ("o", "و"), ("u", "و"), ("g", "ڨ"), ("p", "ب"), ("v", "ڥ"), ("x", "كس"),
    ("c", "ك"),
]


def deromanize(text: str) -> str:
    """Rule-based Arabizi -> Arabic script. Short vowels inside words are dropped as in
    unvocalised Arabic only for 'e'; other vowels map to long-vowel letters. Tokens that
    look French (common function words) are left in Latin script."""
    french = {"le", "la", "les", "de", "des", "et", "est", "un", "une", "pour", "pas",
              "mais", "trop", "tres", "très", "bien", "merci", "avec", "c'est", "je"}
    out = []
    for tok in normalize_arabizi(text).split():
        if tok in french or not re.search(r"[a-z0-9]", tok):
            out.append(tok)
            continue
        i, res = 0, []
        while i < len(tok):
            for src, tgt in _DEROM:
                if tok.startswith(src, i):
                    res.append(tgt)
                    i += len(src)
                    break
            else:
                res.append(tok[i])
                i += 1
        out.append("".join(res))
    return " ".join(out)


# Maghrebi negation (evaluation-only contrast set). Circumfix ma-...-ch/-sh on the verb;
# "mich"/"mech"/"machi" before nominal predicates. The rule negates the first token that
# looks like a verb (heuristic: >= 3 letters, not a known particle).
_PARTICLES = {"w", "wa", "fi", "f", "min", "mn", "3la", "ala", "l", "el", "il", "ya",
              "ana", "enti", "inti", "howa", "hiya", "hna", "rani", "raho"}


def negate_arabizi(text: str) -> str | None:
    toks = normalize_arabizi(text).split()
    for i, t in enumerate(toks):
        if t.startswith("ma") and (t.endswith("ch") or t.endswith("sh")):
            return None  # already negated: skip rather than double-negate
        if t in {"mich", "mech", "machi", "mouch", "moch"}:
            return None
    for i, t in enumerate(toks):
        # \d{2}: user handles / numbers ("m5abbi2163"), not verbs
        if t not in _PARTICLES and len(t) >= 3 and t.isalnum() and not re.search(r"\d{2}", t):
            toks[i] = f"ma{t}ch"
            return " ".join(toks)
    return None


def negate_arabic(text: str) -> str | None:
    toks = normalize_arabic(text).split()
    for t in toks:
        if (t.startswith("ما") and t.endswith("ش")) or t in {"مش", "ماشي", "موش"}:
            return None
    ar_particles = {"و", "في", "من", "على", "يا", "انا", "هو", "هي", "احنا"}
    for i, t in enumerate(toks):
        if t not in ar_particles and len(t) >= 3:
            toks[i] = f"ما{t}ش"
            return " ".join(toks)
    return None


_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")


def strip_emoji(text: str) -> str:
    """Remove emoji and pictographs. MYC has emoji in 81-93% of rows, which carry
    sentiment independently of script and so confound cross-script evaluation."""
    return re.sub(r"\s+", " ", _EMOJI.sub(" ", text)).strip()
