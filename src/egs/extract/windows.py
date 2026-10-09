"""Narrows a transcript down to the passages that can contain guidance.

Feeding a whole call to the model doesn't work. A transcript runs about 12k
tokens, guidance occupies a few hundred of them near the end of the prepared
remarks, and a small model handed the whole thing reports no guidance at all --
measurably: gemma3:4b returns `no_guidance_given` on a full NVDA call and
extracts all eight figures from the same call's outlook paragraph.

So the pipeline extracts per window and merges. Windows come from explicit
outlook cues where they exist, and from overlapping chunks where they don't,
which keeps recall on companies whose CFO never says a recognizable cue phrase.
"""

import re

from egs.schemas import Transcript

MAX_WINDOW_CHARS = 4000
OVERLAP_CHARS = 400

# Each window costs a model call. Cue windows are nearly always the right
# passages, so the cap only really bites on the no-cue fallback, where chunks
# are ranked and the thinnest ones dropped rather than taken in file order.
MAX_WINDOWS = 6

# Guidance almost always follows one of these. Ordered loosely by how
# unambiguous they are, though all of them are matched the same way.
_CUES = (
    r"turn(?:ing)?\s+to\s+(?:our\s+|the\s+)?outlook",
    r"let\s+me\s+turn\s+(?:to\s+)?the\s+outlook",
    r"outlook\s+for\s+(?:the\s+)?(?:first|second|third|fourth|full|fiscal|next)",
    r"(?:our|the)\s+guidance\s+for",
    r"guidance\s+for\s+(?:the\s+)?(?:full\s+year|fiscal|first|second|third|fourth)",
    r"(?:raising|lowering|reiterating|updating|reaffirming)\s+(?:our\s+)?(?:full[-\s]year\s+)?guidance",
    r"mov(?:ing|e)\s+(?:on\s+)?to\s+(?:our\s+)?(?:guidance|outlook)",
    r"now\s+(?:let\s+me|I'?ll|I\s+will)\s+(?:turn\s+to|discuss|cover)\s+(?:our\s+)?(?:guidance|outlook)",
    r"revenue\s+is\s+expected\s+to\s+be",
    r"we\s+expect\s+revenue\s+(?:to\s+be|of|between)",
    # Microsoft guides each segment under its own heading and then returns to
    # company-level figures; neither heading says "outlook".
    r"(?:next\s+)?to\s+segment\s+guidance",
    r"back\s+to\s+(?:company|total)\s+guidance",
    # Apple introduces each line item this way instead of naming the metric
    # first ("For OpEx, we expect to be between $9.8 billion and $9.9 billion").
    r"we\s+expect\s+to\s+be\s+between",
    r"for\s+the\s+(?:first|second|third|fourth)\s+quarter[^.]{0,60}?we\s+expect",
)
_CUE_PATTERN = re.compile("|".join(_CUES), re.IGNORECASE)

# A chunk is only worth a model call if it talks about the future and contains
# a number. Without this the no-cue fallback sends the entire call through in
# pieces, which is slow and mostly reads back empty.
_FORWARD_LOOKING = re.compile(
    r"\b(?:expect|anticipate|guidance|outlook|forecast|project(?:ing|ed)?|"
    r"we\s+see|will\s+be|plan\s+to)\b",
    re.IGNORECASE,
)
_HAS_FIGURE = re.compile(r"\d")

# How much to reach back before a cue, so the sentence introducing the period
# the guidance applies to comes along with it.
_LEAD_CHARS = 250


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse overlapping or touching spans so no passage is sent twice."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _clamp(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Break any span longer than one window back down to window size.

    Merging is what makes this necessary: two overlapping 4000-character spans
    merge into something longer than 4000, and a call that says "we expect"
    every other sentence merges into a span covering most of the transcript --
    which would hand the model the whole call again and defeat the point.
    """
    out: list[tuple[int, int]] = []
    step = MAX_WINDOW_CHARS - OVERLAP_CHARS
    for start, end in spans:
        if end - start <= MAX_WINDOW_CHARS:
            out.append((start, end))
            continue
        for sub in range(start, end, step):
            sub_end = min(end, sub + MAX_WINDOW_CHARS)
            # A span barely over the cap splits into a full window plus a
            # sliver the overlap already covered. That sliver is a whole model
            # call for a few hundred new characters, so drop it.
            if out and sub_end - out[-1][1] < OVERLAP_CHARS:
                continue
            out.append((sub, sub_end))
            if sub_end >= end:
                break
    return out


def _cue_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for match in _CUE_PATTERN.finditer(text):
        start = max(0, match.start() - _LEAD_CHARS)
        spans.append((start, min(len(text), start + MAX_WINDOW_CHARS)))
    return _clamp(_merge_spans(spans))


def _chunk_spans(text: str) -> list[tuple[int, int]]:
    """Overlapping chunks that could hold a forecast, densest first.

    The overlap matters: a guidance sentence split across a chunk boundary
    would otherwise be truncated in both halves and extracted from neither.

    Ranking is by how many forward-looking phrases a chunk contains, so when
    MAX_WINDOWS trims the list it keeps the passages most likely to be the
    outlook section instead of whichever chunks happen to come first.
    """
    scored = []
    step = MAX_WINDOW_CHARS - OVERLAP_CHARS
    for start in range(0, max(1, len(text)), step):
        end = min(len(text), start + MAX_WINDOW_CHARS)
        chunk = text[start:end]
        hits = len(_FORWARD_LOOKING.findall(chunk))
        if hits and _HAS_FIGURE.search(chunk):
            scored.append((hits, start, end))
        if end == len(text):
            break

    scored.sort(key=lambda s: -s[0])
    return sorted((start, end) for _, start, end in scored[:MAX_WINDOWS])


def guidance_windows(transcript: Transcript) -> list[str]:
    """Passages of the call worth sending to the model, in order.

    Prepared remarks are searched for cues first, since that is where
    management reads the numbers out. The Q&A is only mined by cue -- analysts
    ask about guidance constantly and chunking all of it produces a lot of
    calls that return nothing.

    A transcript shorter than one window comes back as a single window, which
    is the same thing the pipeline used to do before windowing existed.
    """
    prepared = transcript.prepared_remarks or ""
    qa = transcript.qa or ""

    if len(prepared) + len(qa) <= MAX_WINDOW_CHARS:
        combined = f"{prepared}\n\n{qa}".strip()
        return [combined] if combined else []

    windows = [prepared[s:e] for s, e in _cue_spans(prepared)]
    if not windows:
        windows = [prepared[s:e] for s, e in _chunk_spans(prepared)]

    windows.extend(qa[s:e] for s, e in _cue_spans(qa))

    # When the Q&A boundary lands at the very start of the call, everything
    # ends up in `qa` and the two passes above have nothing to work with. Chunk
    # the whole call rather than reporting a 38k-character transcript as having
    # no passage worth reading.
    if not windows:
        combined = f"{prepared}\n\n{qa}"
        windows = [combined[s:e] for s, e in _chunk_spans(combined)]

    # A call really can be guidance-free; the caller treats an empty list as
    # no_guidance_given rather than as a failure.
    return windows[:MAX_WINDOWS]
