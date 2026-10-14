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
    "first question will come from",
    "first question will be from",
    "our first question from",
    "we will now open the floor",
    "we'll now open the floor",
    "open the line for your questions",
    "may we have the first question",
    "may we please have the first question",
    "may have the first question",
    "let's open the call to questions",
    "open the line for questions",
    "open it up for questions",
    # Tesla takes written questions from retail and institutional holders
    # before analysts, so its hand-off is the company's own line rather than
    # an operator's.
    "now take questions from",
    "take a retail question",
    "retail questions from",
    "go to investor questions",
    "go to analyst questions",
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
# The determiner list matters: Walmart's IR host says "that will be followed
# by *our* question-and-answer session", and allowing only "a" let that
# forward reference through as if it were the hand-off.
_FORWARD_REFERENCE_BEFORE = re.compile(
    r"(there(?:'ll|’ll| will)\s*be|will (?:be|conduct|hold|have|open)|followed by)"
    r"\s*(?:a|an|our|the|its|this)?\s*$",
    re.IGNORECASE,
)
_FORWARD_REFERENCE_AFTER = re.compile(
    r"^\s*(will follow|is to follow|to follow)", re.IGNORECASE
)


def _straighten(text: str) -> str:
    """Swap curly apostrophes for straight ones so the cue list matches both.

    Six of fourteen Apple transcripts use U+2019, which meant "we'll now begin
    the question" silently missed them. The replacement is one character for
    one character, so match offsets stay valid against the original text.
    """
    return text.replace("\u2019", "'")


def _strong_match(text: str) -> re.Match | None:
    return _STRONG_PATTERN.search(_straighten(text))


def _weak_match(text: str) -> re.Match | None:
    """A weak cue, unless it's framed as a forward reference to the agenda.

    "there will be a Q&A session" and "a Q&A session will follow" preview the
    call's format in the operator's opening; they are not the hand-off.
    """
    straightened = _straighten(text)
    match = _WEAK_PATTERN.search(straightened)
    if not match:
        return None
    before = straightened[max(0, match.start() - 40) : match.start()]
    after = straightened[match.end() : match.end() + 30]
    if _FORWARD_REFERENCE_BEFORE.search(before) or _FORWARD_REFERENCE_AFTER.search(after):
        return None
    return match


def _find_cue(text: str) -> re.Match | None:
    """First cue match in `text`, or None. Strong cues win."""
    return _strong_match(text) or _weak_match(text)


_TITLE_SPLIT = re.compile(r"\s*,\s*|\s+-\s+")

# Paragraph-level boilerplate: operator housekeeping and the safe-harbor legal
# disclaimer. Usually these are whole self-contained paragraphs, but not
# always -- see MAX_BOILERPLATE_CHARS.
# A paragraph longer than this is a presentation that happens to contain a
# disclaimer, not a disclaimer. Observed boilerplate paragraphs sit around 400
# characters; the prepared-remarks turns this used to eat were 20k.
MAX_BOILERPLATE_CHARS = 1500

# A weak cue is only believed when the turns before it hold a plausible amount
# of prepared remarks. Walmart's host intro runs about 1.3k characters and the
# operator's housekeeping a few hundred, while the executives' remarks that
# follow run tens of thousands -- so a weak boundary landing that early is
# mis-splitting the call, whatever the phrasing. Strong cues are unambiguous
# hand-offs and stay trusted wherever they appear.
MIN_PREPARED_FOR_WEAK_CUE = 2000

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


def _boundary_turn(turns: list[tuple[str, str]]) -> int | None:
    """Index of the turn that opens the Q&A, or None if there isn't one.

    A strong cue anywhere outranks a weak cue, rather than whichever happens
    to come first. Boeing's operator describes the call's format up front
    ("plus the analyst question-and-answer session, are being broadcast live"),
    which trips a weak cue on turn 0 even though the real hand-off is 40k
    characters later -- taking the first match of any strength put the entire
    call in `qa` and left `prepared_remarks` empty on every Boeing quarter.

    Turn 0 is never the boundary for the same reason: that turn is the
    operator's housekeeping, and a real call always has prepared remarks
    before its Q&A.
    """
    weak: int | None = None
    running = 0
    for i, (_, text) in enumerate(turns):
        if i == 0:
            running += len(text)
            continue
        if _strong_match(text):
            return i
        if (
            weak is None
            and running >= MIN_PREPARED_FOR_WEAK_CUE
            and _weak_match(text)
        ):
            weak = i
        running += len(text)
    return weak


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
        boundary = _boundary_turn(turns)
        if boundary is not None:
            prepared = "\n\n".join(t for _, t in turns[:boundary])
            qa = "\n\n".join(t for _, t in turns[boundary:])
            return prepared, qa
        return "\n\n".join(t for _, t in turns), ""

    return _regex_split(content)


def _regex_split(content: str | None) -> tuple[str, str]:
    if not content:
        return "", ""

    # A boundary at offset 0 means the cue was in the operator's opening
    # housekeeping, not the hand-off -- same situation `_boundary_turn` guards
    # against, and it would leave prepared_remarks empty. Keep looking past it.
    search_from = 0
    while True:
        match = _find_cue(content[search_from:])
        if not match:
            return content, ""
        start = search_from + match.start()
        # back up to the start of the paragraph the cue sits in, so we don't
        # cut the sentence that opens the Q&A in half
        boundary = content.rfind("\n", 0, start)
        boundary = boundary + 1 if boundary != -1 else 0
        if boundary > 0:
            return content[:boundary].strip(), content[boundary:].strip()
        search_from = start + match.end() - match.start()


def strip_boilerplate(text: str) -> str:
    """Drop operator housekeeping and safe-harbor paragraphs from `text`.

    Only short paragraphs are eligible. A speaker turn is one paragraph here,
    and some CFOs read the safe-harbor sentence and the entire presentation as
    a single turn -- Costco's does, in a 19,952-character paragraph. Matching
    on the disclaimer and dropping the paragraph deleted every word of the
    prepared remarks on three of its four calls, and prepared remarks are
    where guidance lives. Real boilerplate here runs a few hundred characters.

    """
    if not text:
        return text
    paragraphs = re.split(r"\n{2,}", text)
    kept = [
        p
        for p in paragraphs
        if len(p) > MAX_BOILERPLATE_CHARS
        or not any(pat.search(p) for pat in _BOILERPLATE_PATTERNS)
    ]
    return "\n\n".join(kept).strip()
