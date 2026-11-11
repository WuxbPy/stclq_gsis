"""Reference completeness and accuracy audit (pre-submission / CI).

The script answers four questions; any "no" is reported with a nonzero exit code:

1. **Is the metadata right**: for every entry in ``references.bib`` with a DOI, do the
   first author surname, year, volume, issue and pages match the Crossref record (the
   year passes if it matches the print year, the issue year or the online year, because
   this manuscript cites by **issue year**, e.g. Cromley et al., online 2013, printed 2014).
2. **Are there dangling citations**: does every author-year citation in the text have a
   matching entry in the bib.
3. **Are there zombie entries**: is every entry in the bib cited at least once in the text.
4. **Are author-year keys unique**: do two different works produce the **same** citation
   string (a risk specific to author-year, e.g. both ``Williams et al., 2009``), which
   leaves the in-text citation ambiguous.

Why the hand-written list in the template is no longer compared
-----------------------------------------------------------------------
GA requires the reference list in **alphabetical order by author** (unnumbered) with
author-year citations in the text, so ``## References`` is now generated from
``references.bib`` by ``make_refs.py`` as a single source.
"Two maintenance paths drifting apart" is structurally ruled out now (Cromley 2014 used
to be 138–148 in the bib but 132–141 in the hand-written list). The script accordingly
moved from reconciling lists to reconciling the citation graph.

    python results/ga/verify_refs.py            # online check + citation graph
    python results/ga/verify_refs.py --offline  # citation graph only (no network)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Submission package dir: resolved by candidate name; a rename edits only this line.
PAPER = next(
    (p for p in (ROOT / "manuscript",
                 ROOT / "papers" / "Geographical Analysis")
     if p.is_dir()),
    ROOT / "manuscript",
)
BIB = PAPER / "references.bib"
RENDERED = PAPER / "GA_manuscript.md"

sys.path.insert(0, str(HERE))
import make_refs  # noqa: E402

UA = "stclq-reference-checker/1.0 (mailto:noreply@example.org)"


# ----------------------------------------------------------------- Crossref
def crossref(doi: str, timeout: int = 20) -> dict | None:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["message"]
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def norm_pages(s) -> str:
    return re.sub(r"\s*[-–—]{1,2}\s*", "-", str(s or "").strip())


def _years(m: dict) -> set[str]:
    """Candidate publication years: online, print, issue, issued. A citation by
    **issue year** takes whichever of the first three matches."""
    ys = set()
    for field in ("published-print", "published-online", "issued", "published"):
        dp = (m.get(field) or {}).get("date-parts") or []
        if dp and dp[0] and dp[0][0]:
            ys.add(str(dp[0][0]))
    ji = (m.get("journal-issue") or {}).get("published") or {}
    dp = ji.get("date-parts") or []
    if dp and dp[0] and dp[0][0]:
        ys.add(str(dp[0][0]))
    return ys


def check_meta(e: dict) -> list[str]:
    doi = e.get("doi")
    if not doi:
        return []
    m = crossref(doi)
    if m is None:
        return [f"[unresolved] DOI {doi} not found in Crossref"]
    out = []
    ys = _years(m)
    ovr = ISSUE_YEAR_OVERRIDE.get(e.get("_key"))
    if ovr:
        ys.add(ovr)
    if e.get("year") and ys and str(e["year"]) not in ys:
        out.append(f"[year] bib={e['year']} crossref={sorted(ys)}")
    for field, key in (("volume", "volume"), ("number", "issue")):
        cv, ev = m.get(key), e.get(field)
        if cv and ev and str(cv) != str(ev):
            out.append(f"[{field}] bib={ev} crossref={cv}")
    cv, ev = norm_pages(m.get("page")), norm_pages(e.get("pages"))
    # Crossref lists only the **first page** for some old JSTOR records (e.g. ``588`` for
    # ASR 1979, actually 588–608). A single Crossref page that equals the start of the bib
    # range counts as a match, so it is not reported as an error.
    single_prefix = (cv and ev and "-" not in cv and "-" in ev
                     and ev.split("-")[0] == cv)
    if cv and ev and cv != ev and not single_prefix and not ev.startswith("Article"):
        out.append(f"[pages] bib={ev} crossref={cv}")
    ca = (m.get("author") or [{}])[0].get("family", "")
    ea = make_refs.first_surname(e.get("author", ""))
    if ca and ea and ca.split()[-1].lower() not in ea.lower():
        out.append(f"[author1] bib={ea} crossref={ca}")
    return out


# --------------------------------------------------------- citation graph (author-date)
def canon(cite: str) -> str:
    """**Normalise** one citation into a comparable canonical string.

    Why it is needed: GA in-text citations read ``(Smith 1979)`` (no comma between author
    and year, two authors joined by ``and``), while the reference list side may write
    ``Smith & Jones, 1979``. The two sides differ in punctuation and connectives but
    agree in meaning, so unify them before comparing: ``&`` to ``and``, drop all commas,
    collapse whitespace.
    """
    s = cite.replace("&", " and ")
    s = re.sub(r"\s*,\s*", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def accepted_forms(entry: dict) -> set[str]:
    """The citation forms **accepted** for an entry in the text (after normalisation)."""
    return {canon(k) for k in make_refs.intext_keys(entry)}


# Author run: take the longest capitalised run before the year, then match known citation
# forms **by suffix**. This parses institutional authors ("UK Home Office") and leading
# sentence words ("Since Leslie & Kronfeld") alike: the former matches whole, the latter
# matches through the suffix "Leslie & Kronfeld", and neither reports a false positive.
_SUR = r"[A-Z][A-Za-z'\u2019\-]*"
_TOK = rf"(?:{_SUR}|et|al\.|&|and)"
_RUN = rf"{_TOK}(?:(?:\s+|,\s*){_TOK})*"
RUN_C = f"({_RUN})"          # must be an explicit group, else the year lands in group(1)

# Crossref records the **issue year** incompletely for MDPI journals (online date only),
# while MDPI itself cites by issue year. This manuscript cites the year a reader sees on
# the article, so such exceptions are registered explicitly to avoid false positives.
ISSUE_YEAR_OVERRIDE = {"zhou2024detecting": "2024"}


def _suffix_cands(run: str, year: str) -> list[str]:
    """Split the author run into tokens and build one candidate per suffix (the suffix
    absorbs institutional names and leading sentence words)."""
    toks = [t for t in re.split(r"\s+", run.strip()) if t]
    return [canon(f"{' '.join(toks[i:])} {year}") for i in range(len(toks))]


def extract_citations(body: str) -> list[tuple[str, str]]:
    """Extract (author run, year) candidates; a run may carry noise such as a leading
    sentence word, which suffix matching filters out."""
    b = re.sub(r"\s+", " ", body)
    runs: list[tuple[str, str]] = []
    for grp in re.findall(r"\(([^()]{2,400}?)\)", b):
        if not re.search(r"\b(?:19|20)\d{2}\b", grp):
            continue
        for part in grp.split(";"):
            for m in re.finditer(RUN_C + r"\s*,?\s*((?:19|20)\d{2})", part):
                runs.append((m.group(1), m.group(2)))
    for m in re.finditer(RUN_C + r"\s*\(((?:19|20)\d{2})\)", b):
        runs.append((m.group(1), m.group(2)))
    return runs


def check_citations(entries: dict) -> tuple[int, int]:
    if not RENDERED.exists():
        print(f"\n[skip] rendered manuscript {RENDERED.name} not found, run render_manuscript.py first")
        return 0, 0
    body = RENDERED.read_text(encoding="utf-8")
    i = body.find("## References")
    if i > 0:
        body = body[:i]                       # the reference list itself is not scanned
    runs = extract_citations(body)

    accepted: dict[str, list[str]] = {}
    for k, e in entries.items():
        for f in accepted_forms(e):
            accepted.setdefault(f, []).append(k)

    hits: set[str] = set()
    unexplained: list[tuple[str, str]] = []
    for run, yr in runs:
        matched = [c for c in _suffix_cands(run, yr) if c in accepted]
        if matched:
            hits.update(matched)
        else:
            unexplained.append((run, yr))

    zombies = sorted(k for k, e in entries.items()
                     if not (accepted_forms(e) & hits))

    print(f"\nin-text citation candidates: {len(runs)} (distinct strings: {len(hits)}); bib entries: {len(entries)}")
    n_err = 0
    if unexplained:
        print(f"\n!! in-text citations with no bib entry ({len(unexplained)}) -> dangling citations:")
        for run, yr in unexplained:
            print(f"   - {run}, {yr}")
        n_err += len(unexplained)
    else:
        print("OK dangling citations: none")
    if zombies:
        print(f"\n!! bib entries never cited in the text ({len(zombies)}) -> zombie entries:")
        for k in zombies:
            print(f"   - {k}  ({make_refs.intext_cite(entries[k])})")
        n_err += len(zombies)
    else:
        print("OK zombie entries: none")
    return n_err, len(hits)


def check_collisions(entries: dict) -> int:
    """Risk specific to author-year: two different works produce the same citation string."""
    seen: dict[str, list] = {}
    for k, e in entries.items():
        seen.setdefault(make_refs.intext_cite(e), []).append(k)
    dup = {s: ks for s, ks in seen.items() if len(ks) > 1}
    if dup:
        print(f"\n!! citation string collisions ({len(dup)} groups) -> in-text citation ambiguous:")
        for s, ks in dup.items():
            print(f"   - {s}  ← {', '.join(ks)}")
        return len(dup)
    print("OK citation uniqueness: no collisions")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip the online Crossref check")
    a = ap.parse_args()

    entries = make_refs.parse_bib(BIB)
    print(f"bib: {BIB.name}, {len(entries)} entries")
    dupdoi = make_refs.duplicate_dois(entries)
    for x, y, d in dupdoi:
        print(f"  !! {x} and {y} share DOI {d}")

    n_err = len(dupdoi)
    n_nodoi = 0
    if not a.offline:
        print("\n--- 1. Crossref metadata check ---")
        for k, e in sorted(entries.items()):
            probs = check_meta(e)
            if not e.get("doi"):
                n_nodoi += 1
                print(f" -- {k:28s} no DOI (monograph/software/data entries, expected)")
                continue
            print(f" {'OK' if not probs else '!!'} {k:28s} {probs if probs else ''}")
            n_err += len(probs)
        print(f"({n_nodoi} entries without a DOI, not checked online)")

    print("\n--- 2. citation string uniqueness ---")
    n_err += check_collisions(entries)
    print("\n--- 3. citation graph reconciliation (text <-> bib) ---")
    e2, _ = check_citations(entries)
    n_err += e2

    print(f"\ntotal: {n_err} problems")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
