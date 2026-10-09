"""Splits a raw transcript into prepared remarks and Q&A, and trims boilerplate.

Call structure is consistent enough across providers that a handful of phrases
the operator uses to open the floor ("we'll now begin the question-and-answer
session", "take our first question", ...) reliably mark the Q&A boundary. When
we don't have speaker-tagged turns to search, we fall back to the same cues
against the raw text.
"""

import re

# Cues that only ever show up as the actual hand-off to Q&A.
_STRONG_CUES = (
    "we'll now begin the question",
    "we will now begin the question",
    "take our first question",
    "the floor is open",
    "first question comes from",
    "we will now open the floor",
    "we'll now open the floor",
    "open the line for your questions",
    "may we have the first question",
)

# "...session" on its own also shows up in the operator's opening housekeeping
# ("there will be a question-and-answer session" previewing the call's
# format), so it's only trusted as a boundary when it isn't in that framing.
_WEAK_CUES = (
    "question-and-answer session",
    "questions-and-answers session",
    "question and answer session",
)

_STRONG_PATTERN = re.compile(
    "(" + "|".join(re.escape(c) for c in _STRONG_CUES) + ")", re.IGNORECASE
)
_WEAK_PATTERN = re.compile(
    "(" + "|".join(re.escape(c) for c in _WEAK_CUES) + ")", re.IGNORECASE
)
_FORWARD_REFERENCE_BEFORE = re.compile(
    r"(there(?:'ll|’ll| will)\s*be|will (?:be|conduct|hold|have|open)|followed by)"
    r"\s*a?\s*$",
    re.IGNORECASE,
)
_FORWARD_REFERENCE_AFTER = re.compile(
    r"^\s*(will follow|is to follow|to follow)", re.IGNORECASE
)


def _find_cue(text: str) -> re.Match | None:
    """First cue match in `text`, or None. Strong cues win; a weak cue only
    counts if it isn't framed as a forward reference to the call's agenda
    ("there will be a Q&A session" / "a Q&A session will follow")."""
    match = _STRONG_PATTERN.search(text)
    if match:
        return match
    match = _WEAK_PATTERN.search(text)
    if not match:
        return None
    before = text[max(0, match.start() - 40) : match.start()]
    after = text[match.end() : match.end() + 30]
    if _FORWARD_REFERENCE_BEFORE.search(before) or _FORWARD_REFERENCE_AFTER.search(after):
        return None
    return match


_TITLE_SPLIT = re.compile(r"\s*,\s*|\s+-\s+")

# Paragraph-level boilerplate: operator housekeeping and the safe-harbor legal
# disclaimer. These are reliably whole, self-contained paragraphs/turns on
# every call in this dataset, so dropping a matching paragraph outright is
# safe — it never shares a paragraph with real guidance.
_BOILERPLATE_PATTERNS = (
    re.compile(r"ladies and gentlemen.{0,80}welcome to", re.IGNORECASE | re.DOTALL),
    re.compile(r"all participants (are|will be) in a listen-only mode", re.IGNORECASE),
    re.compile(r"press\s+\*?\s*\d\s*(on your telephone)?\s*to ask", re.IGNORECASE),
    re.compile(r"safe harbor", re.IGNORECASE),
    re.compile(
        r"forward-looking statements?.{0,200}(differ materially|risks and uncertainties)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"this (concludes|does conclude) (today's|the) .{0,40}call", re.IGNORECASE),
    re.compile(r"you may (now )?disconnect", re.IGNORECASE),
)


def speaker_turns(structured_content: list[dict] | None) -> list[tuple[str, str]]:
    """Normalize (speaker, text) pairs, stripping titles off the speaker name."""
    if not structured_content:
        return []
    turns = []
    for turn in structured_content:
        raw_speaker = (turn.get("speaker") or "").strip()
        text = (turn.get("text") or "").strip()
        name = _TITLE_SPLIT.split(raw_speaker, maxsplit=1)[0].strip()
        turns.append((name, text))
    return turns


def split_call(
    structured_content: list[dict] | None, content: str | None
) -> tuple[str, str]:
    """Split a transcript into (prepared_remarks, qa).

    Prefers structured_content turns; falls back to regex-splitting the raw
    content when turns aren't available. Returns (everything, "") if no
    boundary is found either way.
    """
    turns = speaker_turns(structured_content)
    if turns:
        for i, (_, text) in enumerate(turns):
            if _find_cue(text):
                prepared = "\n\n".join(t for _, t in turns[:i])
                qa = "\n\n".join(t for _, t in turns[i:])
                return prepared, qa
        return "\n\n".join(t for _, t in turns), ""

    return _regex_split(content)


def _regex_split(content: str | None) -> tuple[str, str]:
    if not content:
        return "", ""
    match = _find_cue(content)
    if not match:
        return content, ""
    # back up to the start of the paragraph the cue sits in, so we don't cut
    # the sentence that opens the Q&A in half
    boundary = content.rfind("\n", 0, match.start())
    boundary = boundary + 1 if boundary != -1 else 0
    return content[:boundary].strip(), content[boundary:].strip()


def strip_boilerplate(text: str) -> str:
    """Drop operator housekeeping and safe-harbor paragraphs from `text`."""
    if not text:
        return text
    paragraphs = re.split(r"\n{2,}", text)
    kept = [p for p in paragraphs if not any(pat.search(p) for pat in _BOILERPLATE_PATTERNS)]
    return "\n\n".join(kept).strip()
