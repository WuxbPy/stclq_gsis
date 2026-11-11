"""Render the reference list for Geographical Analysis from ``references.bib``.

Why this script exists
----------------------
``## References`` used to be **hand-written** in the template, giving two separately
maintained paths to the same list alongside ``references.bib``, and the two drifted:
Cromley et al. (2014) spans 138–148 in the bib but 132–141 in the template list, two
page ranges for one article. Generating the list from the bib as a single source makes
that class of error structurally impossible.

GA format requirements (official Author Guidelines, plus published-article style)
---------------------------------------------------------------------------------
* In-text citations are **author and year**, with **no comma** between the two:
  ``(Smith 1979)``, ``(Smith 1979, p. 241)``; two authors are joined by ``and``
  (not ``&``).
* The reference list is **alphabetical by author** (unnumbered); each entry runs
  surname, initials, **year in parentheses**, title, journal, volume, pages.
* Only the **first author** is inverted (``Cliff, A. D., and J. K. Ord.``); the rest
  keep natural order.
* Journal page ranges are abbreviated Chicago style (``1279–304``).

Usage
-----
    python results/ga/make_refs.py            # print the rendered list
    python results/ga/make_refs.py --write    # write refs/references_list.md
"""

from __future__ import annotations

import argparse
import pathlib
import re
import unicodedata

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Submission package directory: resolved by candidate name; a rename edits one line.
PAPER = next(
    (p for p in (ROOT / "manuscript",
                 ROOT / "papers" / "Geographical Analysis")
     if p.is_dir()),
    ROOT / "manuscript",
)
BIB = PAPER / "references.bib"
OUT = HERE / "refs" / "references_list.md"

# --------------------------------------------------------------------------
# BibTeX parsing (a brace-aware implementation that suffices; no bibtexparser)
# --------------------------------------------------------------------------


_ACCENT = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303"}


def _accents(s: str) -> str:
    r"""Turn ``\'e`` / ``{\"o}`` back into real accented characters (é, ö) rather than
    dropping the accent."""
    def sub(m):
        base, mark = m.group(2), _ACCENT[m.group(1)]
        return unicodedata.normalize("NFC", base + mark)
    s = re.sub(r"\\?([\'`^\"~])\{([A-Za-z])\}", sub, s)   # {\"o}
    s = re.sub(r"\\([\'`^\"~])([A-Za-z])", sub, s)         # \"o
    return s


def _strip_braces(s: str) -> str:
    """Strip BibTeX's protective braces and restore LaTeX escapes to Unicode."""
    s = _accents(s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("---", "—").replace("--", "–")
    s = s.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
    return re.sub(r"\s+", " ", s).strip()


def _split_fields(body: str):
    """Split ``a = {..}, b = {..}`` into (field, value); values may nest braces."""
    out = []
    i, n = 0, len(body)
    while i < n:
        m = re.compile(r"([A-Za-z]+)\s*=\s*").search(body, i)
        if not m:
            break
        name = m.group(1).lower()
        j = m.end()
        if body[j] == "{":
            depth, k = 0, j
            while k < n:
                if body[k] == "{":
                    depth += 1
                elif body[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            val = body[j + 1:k]
            i = k + 1
        else:  # bare value, e.g. year = 2020
            k = j
            while k < n and body[k] not in ",\n":
                k += 1
            val = body[j:k]
            i = k
        out.append((name, val))
    return out


def parse_bib(path=BIB):
    """Return ``{key: {...}}``."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    text = re.sub(r"(?m)^\s*%.*$", "", text)          # drop comment lines
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{", text):
        etype = m.group(1).lower()
        start = m.end()
        depth, k = 1, start
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        body = text[start:k - 1]
        if "," not in body:
            continue
        key, rest = body.split(",", 1)
        fields = {n: _strip_braces(v) for n, v in _split_fields(rest)}
        fields["_type"] = etype
        fields["_key"] = key.strip()
        entries[fields["_key"]] = fields
    return entries


# --------------------------------------------------------------------------
# author-string handling
# --------------------------------------------------------------------------

def author_list(raw: str):
    """``"Leslie, Timothy F. and Kronenfeld, Barry J."`` → ['Leslie, Timothy F.', ...]"""
    if not raw:
        return [], False
    parts = [p.strip() for p in re.split(r"\s+and\s+", raw) if p.strip()]
    et_al = False
    if parts and parts[-1].lower() == "others":
        parts, et_al = parts[:-1], True
    return parts, et_al


def _surname(author: str) -> str:
    return author.split(",")[0].strip() if "," in author else author.strip()


def _initials(author: str) -> str:
    """``"Leslie, Timothy F."`` → ``"TF"``; ``"Getis, Arthur"`` → ``"A"``."""
    if "," not in author:
        return ""
    given = author.split(",", 1)[1].strip()
    letters = re.findall(r"[A-Za-zÀ-ÿ]", given)
    # hyphenated given names (Mei-Po) take the initial of each segment → MP
    segs = [s for s in re.split(r"[-\s]+", given) if s]
    return "".join(s[0].upper() for s in segs) if segs else "".join(letters)


def given_initials(author: str) -> str:
    """Given-name initials with periods: ``"Leslie, Timothy F."`` → ``"T. F."``;
    ``"Ord, J. K."`` → ``"J. K."``.

    Hyphenated given names follow Chicago and become ``M.-P.`` (Mei-Po → ``M.-P.``).
    """
    if "," not in author:
        return ""
    given = author.split(",", 1)[1].strip()
    segs = [s for s in re.split(r"[\s.]+", given) if s]
    out = []
    for s in segs:
        parts = [p for p in s.split("-") if p]
        out.append("-".join(f"{p[0].upper()}." for p in parts))
    return " ".join(out)


def ref_authors(raw: str) -> str:
    """Reference-list author string (GA): the **first author is inverted**, the rest keep
    natural order, and ``and`` precedes the last name.

    ``"Cliff, A. D. and Ord, J. K."`` → ``"Cliff, A. D., and J. K. Ord"``
    Corporate authors (no comma, e.g. ``UK Home Office``) are returned unchanged.
    """
    authors, et_al = author_list(raw)
    if not authors:
        return ""
    first = authors[0]
    fam, ini = _surname(first), given_initials(first)
    parts = [f"{fam}, {ini}".strip().rstrip(",") if ini else fam]
    for a in authors[1:]:
        fam2, ini2 = _surname(a), given_initials(a)
        parts.append(f"{ini2} {fam2}".strip() if ini2 else fam2)
    if et_al:
        return ", ".join(parts) + ", et al."
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]}, and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def short_authors(raw: str, max_listed: int = 2):
    """``Surname II, Surname II`` for the reference list (the caller decides when a long
    list folds to etc)."""
    authors, et_al = author_list(raw)
    parts = [f"{_surname(a)} {_initials(a)}".strip() for a in authors]
    return parts, et_al


def first_surname(raw: str) -> str:
    authors, _ = author_list(raw)
    return _surname(authors[0]) if authors else ""


def intext_cite(entry) -> str:
    """In-text citation string (GA: **no comma** between author and year; up to three
    names are listed, four or more fold to et al.)."""
    authors, et_al = author_list(entry.get("author", ""))
    names = [_surname(a) for a in authors]
    yr = entry.get("year", "n.d.")
    if not names:
        return f"{entry.get('_key', '?')} {yr}"          # corporate author
    # ``and others`` means more authors exist than are listed, so the et al. form is
    # mandatory; otherwise the output would read "Harris, Millman and van der Walt et al.
    # 2020", a form that does not exist.
    if et_al or len(names) >= 4:
        return f"{names[0]} et al. {yr}"
    if len(names) == 1:
        who = names[0]
    elif len(names) == 2:
        who = f"{names[0]} and {names[1]}"
    else:
        who = f"{names[0]}, {names[1]}, and {names[2]}"
    return f"{who} {yr}"


def intext_keys(entry) -> list:
    """**All** citation forms this entry can take in the text, for the completeness check."""
    authors, et_al = author_list(entry.get("author", ""))
    names = [_surname(a) for a in authors]
    yr = str(entry.get("year", "n.d.")).strip()
    keys = set()
    if not names:
        keys.add(f"{entry.get('_key', '?')} {yr}")
        return sorted(keys)
    if et_al or len(names) >= 4:
        keys.add(f"{names[0]} et al. {yr}")
    elif len(names) == 1:
        keys.add(f"{names[0]} {yr}")
    elif len(names) == 2:
        keys.update({f"{names[0]} and {names[1]} {yr}",
                     f"{names[0]}, {names[1]} {yr}",
                     f"{names[0]} & {names[1]}, {yr}"})
    else:
        keys.update({f"{names[0]}, {names[1]}, and {names[2]} {yr}",
                     f"{names[0]}, {names[1]} and {names[2]}, {yr}",
                     f"{names[0]}, {names[1]}, {names[2]} {yr}",
                     f"{names[0]} et al. {yr}"})
    return sorted(keys)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _venue(e) -> str:
    return e.get("journal") or e.get("booktitle") or ""


def pagerange(pages) -> str:
    """Chicago-style abbreviated page range: ``1279--1304`` → ``1279–304``,
    ``93--115`` → ``93–115``.

    Rule (CMS 9.61): ranges of two digits or fewer are written in full; otherwise the
    leading digits the two numbers share are dropped, but the second number **keeps at
    least two digits**. Non-numeric ranges (``Article 39``, article numbers) are
    returned unchanged.
    """
    raw = str(pages or "").strip()
    m = re.match(r"^(\d+)\s*[-–—]{1,2}\s*(\d+)$", raw)
    if not m:
        return raw
    a, b = m.group(1), m.group(2)
    if len(a) != len(b) or len(a) <= 2:
        return f"{a}–{b}"
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    if i == 0:
        return f"{a}–{b}"
    keep = max(2, len(b) - i)
    return f"{a}–{b[len(b) - keep:]}"


def _tail(e) -> str:
    """Tail of ``volume, pages`` (GA uses a comma, not a colon)."""
    vol, pages = e.get("volume"), e.get("pages")
    if vol and pages:
        return f" {vol}, {pagerange(pages)}"
    if vol:
        return f" {vol}"
    if pages:
        return f" {pagerange(pages)}"
    return ""


def ordinal(n) -> str:
    """``2`` → ``2nd`` (for the edition of a book)."""
    try:
        i = int(n)
    except (TypeError, ValueError):
        return str(n)
    suf = "th" if 10 <= i % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suf}"


def format_entry(e) -> str:
    """Render one entry in GA format (year in parentheses, first author inverted, title
    in quotes and, for articles, journal in italics)."""
    head = ref_authors(e.get("author", "")) or e.get("_key", "?")
    yr = e.get("year", "n.d.")
    title = e.get("title", "").rstrip(".")
    t = e.get("_type", "article")
    ed = e.get("edition")
    pg = e.get("pages")

    if t in ("book", "monograph"):
        s = f"{head} ({yr}). *{title}*."
        if ed:
            s += f" {ordinal(ed)} ed."
        if e.get("address") and e.get("publisher"):
            s += f" {e['address']}: {e['publisher']}."
        elif e.get("publisher"):
            s += f" {e['publisher']}."
        elif e.get("address"):
            s += f" {e['address']}."
    elif t == "incollection":
        s = f"{head} ({yr}). “{title}.” In *{_venue(e)}*"
        if pg:
            s += f", {pagerange(pg)}"
        s += "."
        if e.get("publisher"):
            s += f" {e['publisher']}."
    elif t == "inproceedings":
        s = f"{head} ({yr}). “{title}.” In *{_venue(e)}*"
        if pg:
            s += f", {pagerange(pg)}"
        s += "."
    elif t in ("misc", "software", "online"):
        s = f"{head} ({yr}). *{title}*."
        if e.get("howpublished"):
            s += f" {e['howpublished']}."
        if e.get("url"):
            s += f" Available at {e['url']}"
    else:
        s = f"{head} ({yr}). “{title}.” *{_venue(e)}*{_tail(e)}."

    if e.get("note"):
        s += f" {e['note']}."
    if e.get("doi"):
        s += f" DOI: {e['doi']}."
    return re.sub(r"[ \t]+", " ", s).strip()


def _sort_key(e):
    fam = first_surname(e.get("author", "")) or e.get("_key", "")
    ascii_fam = unicodedata.normalize("NFKD", fam).encode("ascii", "ignore").decode()
    return (ascii_fam.lower(), str(e.get("year", "")), e.get("title", "").lower())


def render(entries=None) -> str:
    """Return the reference list sorted alphabetically by author (GA requires
    alphabetical order and no numbering)."""
    entries = entries if entries is not None else parse_bib()
    lines = [format_entry(e) for e in sorted(entries.values(), key=_sort_key)]
    return "\n\n".join(lines)


def duplicate_dois(entries=None):
    """Check whether one DOI is carried by two keys (so no reference appears twice)."""
    entries = entries if entries is not None else parse_bib()
    seen, dup = {}, []
    for k, e in entries.items():
        d = (e.get("doi") or "").lower().strip()
        if not d:
            continue
        if d in seen:
            dup.append((seen[d], k, d))
        seen[d] = k
    return dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write refs/references_list.md")
    args = ap.parse_args()

    entries = parse_bib()
    dup = duplicate_dois(entries)
    print(f"references.bib: {len(entries)} entries; duplicate DOIs: {len(dup)}")
    for a, b, d in dup:
        print(f"  !! {a} and {b} share DOI {d}")
    txt = render(entries)
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(txt + "\n", encoding="utf-8")
        print("wrote ->", OUT)
    else:
        print("-" * 78)
        print(txt)


if __name__ == "__main__":
    main()
