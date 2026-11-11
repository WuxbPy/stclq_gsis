#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qa_manuscript.py: mechanical checks on the rendered manuscript (complements
verify_refs.py).

verify_refs.py covers the reference system (DOIs, citation graph, name clashes); this
script covers the surface quality of the text. Every check is deterministic and offline:

  C1  unfilled template placeholders left behind: ``{{...}}``
  C2  numeric citation marks left behind: ``[1]`` ... ``[99]`` (GA cites author and year)
  C3  repeated 5-grams inside a paragraph (the copy-paste damage that incremental
      editing leaves)
  C4  adjacent repeated words: ``the the`` / ``of of``
  C5  repeated punctuation: ``. .`` / ``, ,`` / ``..`` / ``,,``
  C6  stray Chinese characters (the manuscript is entirely English; this flags CJK only)
  C7  abstract length against the 300-word ceiling of the GSIS structured abstract
  C8  figure and table citation consistency: ``Figure N`` / ``Table N`` / ``Fig. N``
      cited vs defined
  C9  consecutive sentences opening with the same word ("The ... The ..."), advisory only
  C10 trailing or leading whitespace, more than two blank lines in a row
  C11 source file names declared in figure captions (`figN_xxx.pdf`) must exist in
      figures/

Usage:
    python results/ga/qa_manuscript.py            # check the MDS render
    python results/ga/qa_manuscript.py --quiet    # print problems only
Exit code: 0 = no P1/P2 problem; 1 = problems found.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# Submission package directory: resolved by candidate name; a rename edits one line.
PAPER = next(
    (p for p in (ROOT / "manuscript",
                 ROOT / "papers" / "Geographical Analysis")
     if p.is_dir()),
    ROOT / "manuscript",
)
MANUSCRIPT = PAPER / "GA_manuscript.md"

# Abstract requirements of Geo-Spatial Information Science (official Instructions
# for Authors, updated 2026-10-14)
# ---------------------------------------------------------------------------
# The paper moved to GSIS on 2026-11-11, which requires a **structured abstract of
# 300 words**; this manuscript uses the structure Context / Methods / Results /
# Conclusions. The earlier 200-word threshold was a hard validation rule of Wiley's
# submission system (Research Exchange) and lapsed with the change of journal.
ABSTRACT_MAX_WORDS = 300          # GSIS structured abstract ceiling
ABSTRACT_MAX_CHARS = 2600         # for reference only: order of magnitude at 9 pt

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "as", "at", "that", "this",
    "it", "its", "we", "our", "not", "but", "from", "which", "can",
}

# C3 allow list: **deliberate** repetition inside a paragraph, not copy-paste damage.
# The key is the matched fragment, the value the reason.
C3_ALLOW: dict[str, str] = {
    # §6.5 deliberately contrasts two permutation nulls: the parallel wording is
    # rhetoric, not redundancy
    "therefore tests type is independent": "§6.5 parallel structure of the two nulls",
}

issues: list[tuple[str, str, int, str]] = []   # (level, code, line, message)


def add(level: str, code: str, line: int, msg: str) -> None:
    issues.append((level, code, line, msg))


# --------------------------------------------------------------------------
# Body chunking: split the markdown into (line number, paragraph text). A paragraph is a
# run of non-empty lines that are not headings, table rows or display-math blocks.
# --------------------------------------------------------------------------
# Skip region for C3/C4/C5/C6: the instruction block that **must be deleted** before
# submission. It sits on the first page as a block quote but addresses the authors
# rather than being prose; its affiliation table repeats institution names by nature,
# which is not copy-paste damage.
AUTHOR_BLOCK_MARK = "delete this block"


def paragraphs(text: str):
    lines = text.split("\n")
    buf: list[str] = []
    start = 0
    in_fence = False
    in_display = False          # $$ ... $$ display-math block (may span lines)
    in_author_block = False     # "... delete this block." instruction block
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if s.startswith(">"):
            if AUTHOR_BLOCK_MARK in s:
                in_author_block = True
            if in_author_block:
                continue
        else:
            in_author_block = False
        if s.startswith("```"):
            in_fence = not in_fence
            if buf:
                yield start, " ".join(buf)
                buf = []
            continue
        if in_fence:
            continue
        # display math: skip the whole block, or its continuation lines become paragraphs
        disp_marks = s.count("$$")
        if disp_marks:
            if disp_marks % 2 == 1:
                in_display = not in_display
            if buf:
                yield start, " ".join(buf)
                buf = []
            continue
        if in_display:
            continue
        # heading / table / rule / image / bold line not opening with emphasis -> boundary
        boundary = (
            not s
            or s.startswith("#")
            or s.startswith("|")
            or s.startswith("---")
            or s.startswith("![")
            or re.match(r"^\*\*[A-Z]", s) is None and s.startswith("**")
        )
        if boundary:
            if buf:
                yield start, " ".join(buf)
                buf = []
            continue
        if not buf:
            start = i
        buf.append(s)
    if buf:
        yield start, " ".join(buf)


# Break marker: math and LaTeX markup are replaced by it, cutting adjacency and n-grams
# that span formulas
BREAK = "\x00"
# Sentinel: math becomes one non-blank character for the punctuation checks, which blocks
# false cross-formula matches such as `,\s*,`
SENTINEL = "\x01"


def masked_raw(s: str) -> str:
    """Replace math and emphasis with a single sentinel character, keeping the relative
    positions. Used by the punctuation checks.

    Unlike :func:`segments`, this must leave a **non-blank** placeholder, or
    ``$Y\\in\\{0,1\\}$, $Y_{ia}=1$`` would collapse into ``… , , …`` and raise a false
    positive.
    """
    s = re.sub(r"\$\$.*?\$\$", SENTINEL, s, flags=re.S)
    s = re.sub(r"\\\[.*?\\\]", SENTINEL, s, flags=re.S)
    s = re.sub(r"\\\(.*?\\\)", SENTINEL, s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", SENTINEL, s)
    s = re.sub(r"\\(?:mathrm|text|mathbb|mathcal)\{[^{}]*\}", SENTINEL, s)
    s = re.sub(r"\\[a-zA-Z]+\b", SENTINEL, s)
    s = re.sub(r"[{}]", SENTINEL, s)
    s = re.sub(r"[*_`]+", "", s)
    return s


def segments(s: str) -> list[str]:
    """Cut a paragraph into natural-language fragments, with math and LaTeX markup turned
    into break points.

    Math must become a **break**, not a space placeholder: ``x_i x_i`` would fake an
    adjacent repeated word and ``$a$ and $a$`` would fake an n-gram. Once the break is
    cut, those relations disappear by themselves.
    """
    s = re.sub(r"\$\$.*?\$\$", BREAK, s, flags=re.S)
    s = re.sub(r"\\\[.*?\\\]", BREAK, s, flags=re.S)
    s = re.sub(r"\\\(.*?\\\)", BREAK, s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", BREAK, s)
    s = re.sub(r"\\[a-zA-Z]+\b", BREAK, s)     # \mathrm \top \alpha ...
    s = re.sub(r"[{}]", BREAK, s)               # LaTeX grouping braces
    s = re.sub(r"[*_`]+", "", s)                # markdown emphasis
    return [t for t in s.split(BREAK)]


def plain(s: str) -> str:
    """Turn a whole paragraph into plain text (fragments joined by spaces), for the
    abstract word count and similar."""
    return re.sub(r"\s+", " ", " ".join(segments(s))).strip()


# Word shape: starts with a letter; may contain digits, apostrophes, hyphens (Unicode
# dashes included), so "S1" and "log–log" each count as one word
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\u2019\-\u2010-\u2015]*")


def words(s: str) -> list[str]:
    return WORD_RE.findall(s)


def gaps(seg: str) -> list[tuple[str, str]]:
    """Return [(word, the raw separator between it and the previous word)]; the first is
    ""."""
    out: list[tuple[str, str]] = []
    last = None
    for m in WORD_RE.finditer(seg):
        out.append((m.group(0), "" if last is None else seg[last:m.start()]))
        last = m.end()
    return out


# Unfilled template placeholders: only {{UPPER_SNAKE}}, not the `}}` of LaTeX
# `_{\text{...}}`
TOKEN_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")

# --------------------------------------------------------------------------
# Plain-text abstract for submission: the submission form is a plain-text field, so
# `$...$` cannot be pasted in as it stands. The LaTeX to plain-text downgrade below is
# **deterministic** and covers only the notation this abstract actually uses.
# --------------------------------------------------------------------------
_MATH_MACROS = {
    # font and grouping wrappers: keep the body only
    "mathbb": "", "mathrm": "", "mathcal": "", "mathsf": "", "mathit": "",
    "text": "", "textrm": "", "operatorname": "",
    # function names
    "log": "log", "ln": "ln", "exp": "exp", "arg": "arg",
    "max": "max", "min": "min", "sup": "sup", "inf": "inf",
    # symbols
    "top": "\u1d40", "perp": "\u27c2",
    "alpha": "\u03b1", "beta": "\u03b2", "gamma": "\u03b3", "delta": "\u03b4",
    "epsilon": "\u03b5", "theta": "\u03b8", "lambda": "\u03bb", "mu": "\u03bc",
    "nu": "\u03bd", "xi": "\u03be", "pi": "\u03c0", "rho": "\u03c1",
    "sigma": "\u03c3", "tau": "\u03c4", "phi": "\u03c6", "psi": "\u03c8",
    "times": "\u00d7", "cdot": "\u00b7", "pm": "\u00b1",
    "in": "\u2208", "notin": "\u2209", "subset": "\u2282",
    "le": "\u2264", "leq": "\u2264", "ge": "\u2265", "geq": "\u2265",
    "neq": "\u2260", "approx": "\u2248", "equiv": "\u2261",
    "lesssim": "\u2272", "gtrsim": "\u2273",
    "iff": "\u21d4", "to": "\u2192", "propto": "\u221d",
    "infty": "\u221e", "partial": "\u2202", "sum": "\u03a3", "prod": "\u03a0",
    "quad": " ", "qquad": "  ",
    # spacing and escaped characters
    ",": " ", ";": " ", "!": "", ":": " ", " ": " ",
}
_LATEX_ESC = {"\\{": "{", "\\}": "}", "\\%": "%", "\\&": "&",
              "\\#": "#", "\\_": "_", "\\$": "$"}
# Function-name macros: pad with a space when preceded by an alphanumeric
# ("N\log N" -> "N log N", readable)
_FUNC_MACROS = {"log", "ln", "exp", "arg", "max", "min", "sup", "inf"}

# Match **known** macro names only, longest first; otherwise a greedy [a-zA-Z]+ would
# swallow `\in` together with the following `R` in `\in\mathbb{R}` into `inR`.
_KNOWN_MACROS = sorted((k for k in _MATH_MACROS if k.isalpha()),
                       key=len, reverse=True)
_MACRO_ONLY_RE = re.compile(
    r"\\(?P<name>" + "|".join(re.escape(k) for k in _KNOWN_MACROS) + r")"
    r"(?P<ws>[ \t]*)(?:\{(?P<body>[^{}]*)\})?")
_MACRO_ANY_RE = re.compile(r"\\([a-zA-Z]+)")

# Unknown macros are dropped; the caller reads the list from this module variable and
# warns the user
UNKNOWN_MACROS: list[str] = []


def to_submission_text(s: str) -> str:
    """Downgrade the abstract markdown to pasteable plain text (one paragraph, no line
    breaks).

    Only the notation the abstract really uses is covered; an unknown macro is **dropped
    and logged** (into :data:`UNKNOWN_MACROS`) rather than silently mangled.
    """
    # 0) handle \frac{}{} first: generic brace stripping would break it apart
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)

    # 1) known macros: \cmd{body} -> symbol + body; \cmd -> symbol
    def _sub_known(m: re.Match) -> str:
        name = m.group("name")
        body = m.group("body")
        ws = "" if body is not None else m.group("ws")
        val = _MATH_MACROS[name]
        if name in _FUNC_MACROS:
            prev = s[max(0, m.start() - 1):m.start()]
            if prev.isalnum():
                val = " " + val
        return val + ws + (body or "")

    prev = None
    while prev != s:                       # body may nest, iterate to a fixed point
        prev = s
        s = _MACRO_ONLY_RE.sub(_sub_known, s)

    # 2) unknown macros: log and drop (as a space, not glued onto the following text)
    def _sub_unknown(m: re.Match) -> str:
        if m.group(1) not in UNKNOWN_MACROS:
            UNKNOWN_MACROS.append(m.group(1))
        return " "

    s = _MACRO_ANY_RE.sub(_sub_unknown, s)

    # 3) escaped characters, leftover braces, math delimiters
    for k, v in _LATEX_ESC.items():
        s = s.replace(k, v)
    s = s.replace("{", "").replace("}", "").replace("$", "")

    # 4) markdown emphasis
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)

    # 5) collapse to one paragraph
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--path", default=str(MANUSCRIPT))
    ap.add_argument("--emit-abstract", default=None, metavar="PATH",
                    help="export the abstract as plain text, ready to paste into the submission form")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"[fatal] manuscript not found: {p}")
        return 1
    text = p.read_text(encoding="utf-8")
    lines = text.split("\n")
    n_lines = len(lines)

    # ---- C1 unfilled tokens ----
    for i, ln in enumerate(lines, 1):
        for m in TOKEN_RE.finditer(ln):
            add("P1", "C1", i, f"unfilled template placeholder: {m.group(0)}")

    # ---- C2 numeric citations ----
    for i, ln in enumerate(lines, 1):
        for m in re.finditer(r"\[\d{1,2}(?:\s*[,;]\s*\d{1,2})*\]", ln):
            add("P1", "C2", i, f"numeric citation mark `{m.group(0)}` not converted to author and year")

    # ---- per-paragraph checks ----
    para_list = list(paragraphs(text))
    for start, raw in para_list:
        segs = segments(raw)
        # C4 adjacent repeated words: the two words must have **only whitespace** between
        #    them ("log–log", "R. R" and "crime data, data.police.uk" do not count)
        for seg in segs:
            gs = gaps(seg)
            for j in range(1, len(gs)):
                a, b = gs[j - 1][0], gs[j][0]
                if (a.lower() == b.lower() and gs[j][1].strip() == ""
                        and a.lower() not in {"that", "had"}):
                    add("P1", "C4", start, f"adjacent repeated word: …{a} {b}…")
        # C3 repeated 5-grams within a paragraph (n-grams form inside one fragment only;
        # the window restarts at a math break)
        seen: dict[tuple, int] = {}
        for seg in segs:
            low = [w.lower() for w in words(seg)]
            for j in range(len(low) - 4):
                g = tuple(low[j:j + 5])
                if sum(1 for w in g if w not in STOP) < 4:
                    continue
                if any(len(w) == 1 for w in g):
                    continue
                phrase = " ".join(g)
                if any(k in phrase for k in C3_ALLOW):
                    continue          # known deliberate repetition
                if g in seen:
                    add("P2", "C3", start, f"repeated 5-gram within a paragraph: \"{phrase}\"")
                seen[g] = j
        # C5 repeated punctuation (run on the sentinel-masked text, so nothing matches
        # across a formula)
        for m in re.finditer(r"(?<![A-Za-z0-9])(\.[,;]|,,|;;|\.\s*\.|,\s*,)",
                             masked_raw(raw)):
            add("P2", "C5", start, f"suspicious repeated punctuation: `{m.group(0).strip()}`")
        # C6 Chinese characters
        for m in re.finditer(r"[\u4e00-\u9fff]+", raw):
            add("P1", "C6", start, f"Chinese characters in the text: {m.group(0)[:30]}")

    # ---- line level: trailing whitespace / consecutive blank lines ----
    for i, ln in enumerate(lines, 1):
        if ln != ln.rstrip():
            add("P3", "C10", i, "trailing whitespace")
    run = 0
    for i, ln in enumerate(lines, 1):
        run = run + 1 if not ln.strip() else 0
        if run == 3:
            add("P3", "C10", i, "three or more consecutive blank lines")

    # ---- C7 abstract ----
    # The abstract body runs from `## Abstract` to `**Keywords:` or the next `##` heading.
    # The measure is **the text actually pasted into the submission form** (plain text
    # after the LaTeX downgrade):
    #   · GA sets no limit, so 250 words is a customary reference and overshoot is P2;
    #   · the raw markdown character count is reported as the worst case (someone pasting
    #     `$...$` in unchanged).
    m = re.search(r"^##\s*Abstract\s*$", text, re.M)
    if not m:
        add("P1", "C7", 0, "no `## Abstract` section found")
    else:
        rest = text[m.end():]
        cut = [len(rest)]
        for pat in (r"^\*\*Keywords", r"^##\s", r"^#\s"):
            mm = re.search(pat, rest, re.M)
            if mm:
                cut.append(mm.start())
        abs_txt = rest[:min(cut)].strip().strip("-").strip()
        UNKNOWN_MACROS.clear()
        sub = to_submission_text(abs_txt)
        w = len(words(sub)) + len(re.findall(r"\d[\d,\.]*", sub))
        c = len(sub)
        c_md = len(abs_txt)
        over = w > ABSTRACT_MAX_WORDS or c > ABSTRACT_MAX_CHARS
        uw = w / ABSTRACT_MAX_WORDS
        uc = c / ABSTRACT_MAX_CHARS
        msg = (f"abstract {w} words / {c} characters (GSIS ceiling "
               f"{ABSTRACT_MAX_WORDS} words; usage {uw:.0%})  "
               f"[source markdown {c_md} characters, for reference only]")
        if over:
            add("P1", "C7", 0, "abstract exceeds the 300-word ceiling of the GSIS "
                               "structured abstract: " + msg)
        else:
            if uw >= 0.97:                    # warn within 3% of the hard ceiling
                add("P2", "C7", 0, "abstract close to the submission ceiling (>= 97%): " + msg)
            if not args.quiet:
                print(f"[ok] C7  {msg}")
        if UNKNOWN_MACROS:
            add("P2", "C7", 0, "unrecognised LaTeX macros dropped while downgrading the "
                               "abstract: " + ", ".join("\\" + x for x in UNKNOWN_MACROS))
        if args.emit_abstract:
            out = Path(args.emit_abstract)
            out.parent.mkdir(parents=True, exist_ok=True)
            # The abstract is structured, so it is exported paragraph by paragraph:
            # Context / Methods / Results / Conclusions each stay a block, and the
            # text can be pasted into the submission form without losing the labels.
            paras = [q.strip() for q in re.split(r"\n\s*\n", abs_txt) if q.strip()]
            if len(paras) > 1:
                blocks = [to_submission_text(re.sub(r"\s*\n\s*", " ", q))
                          for q in paras]
                body = "\n\n".join(blocks)
            else:
                body = sub
            out.write_text(body + "\n", encoding="utf-8")
            print(f"[ok] C7  exported a plain-text abstract for submission -> {out} "
                  f"({w} words / {c} characters, {len(body.split(chr(10) * 2))} paragraphs)")

    # ---- C8 figure and table citation consistency ----
    def refs(pat: str) -> set[str]:
        out: set[str] = set()
        for mm in re.finditer(pat, text):
            out.update(re.findall(r"\d+", mm.group(0)))
        return out

    fig_cited = refs(r"(?:Figure|Fig\.)\s*(?:\d+|[A-Z]?\d+)(?:\s*(?:,|and|&)\s*(?:Figure|Fig\.)?\s*\d+)*")
    fig_cited |= refs(r"(?:Figure|Fig\.)\s*(S?\d+)")
    tab_cited = refs(r"Table\s*(?:S?\d+)")
    # a lone "Figure 1" citation must be caught too
    fig_defined = set(re.findall(r"^\*\*Figure\s+(\d+)\b", text, re.M))
    fig_defined |= set(re.findall(r"^!\[Figure\s+(\d+)", text, re.M))
    fig_defined |= set(re.findall(r"^#{2,4}\s*Figure\s+(\d+)", text, re.M))
    for n in sorted(fig_defined - fig_cited, key=lambda x: int(x)):
        add("P2", "C8", 0, f"Figure {n} defined but never cited in the text")
    for n in sorted(fig_cited - fig_defined, key=lambda x: int(x)):
        if int(n) <= 10:
            add("P2", "C8", 0, f"Figure {n} cited but no caption or title found")
    tab_defined = set(re.findall(r"^\*\*Table\s+(\d+)\b", text, re.M))
    tab_defined |= set(re.findall(r"^#{2,4}\s*Table\s+(\d+)", text, re.M))
    for n in sorted(tab_defined - tab_cited, key=lambda x: int(x)):
        add("P2", "C8", 0, f"Table {n} defined but never cited in the text")

    # ---- C11 source file names in figure captions must exist ----
    # When figure numbers shift or files are renamed, the ```figN_xxx.pdf` `` at the end of
    # a caption is the thing most easily missed, and a pointer to a missing file is
    # **not** caught by any other check: the render and the word counts look fine.
    # Here the files declared in the text are reconciled against figures/.
    cited_files = set(re.findall(r"([A-Za-z0-9_]+\.(?:pdf|png|eps|tiff?))", text))
    if cited_files:
        figdir = HERE / "figures"
        on_disk = {q.name for q in figdir.glob("*")} if figdir.is_dir() else set()
        if not on_disk:
            add("P2", "C11", 0, f"figure directory {figdir} not found, cannot check caption source files")
        else:
            for f in sorted(cited_files - on_disk):
                add("P1", "C11", 0,
                    f"source file `{f}` declared in a caption does not exist under figures/ (stale figure number or name?)")
            if not (cited_files - on_disk) and not args.quiet:
                print(f"[ok] C11 all {len(cited_files)} caption source files exist")

    # ---- report ----
    print()
    print("=" * 72)
    print(f"manuscript check: {p.name}   ({n_lines} lines, {len(text):,} characters, "
          f"{len(para_list)} paragraphs)")
    print("=" * 72)
    order = {"P1": 0, "P2": 1, "P3": 2}
    issues.sort(key=lambda t: (order.get(t[0], 9), t[1], t[2]))
    if not issues:
        print("✅ no P1/P2/P3 problem found")
    else:
        for lvl, code, line, msg in issues:
            loc = f"L{line:<5}" if line else "  -   "
            print(f"[{lvl}] {code} {loc} {msg}")
    print("-" * 72)
    n1 = sum(1 for t in issues if t[0] == "P1")
    n2 = sum(1 for t in issues if t[0] == "P2")
    n3 = sum(1 for t in issues if t[0] == "P3")
    print(f"P1 (must fix) {n1}   P2 (advisory) {n2}   P3 (typography) {n3}")
    return 1 if (n1 or n2) else 0


if __name__ == "__main__":
    sys.exit(main())
