"""System prompt and user-prompt builder for guidance extraction.

PROMPT_VERSION feeds into the cache key (see base.LLMProvider.cache_key), so
bump it whenever a wording change could plausibly change model output.
"""

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """\
You extract forward-looking financial guidance from earnings call transcripts.

Only extract guidance management actually gave for a FUTURE period. Never
extract the actual results for the quarter being reported, even if a figure
sounds like a range (e.g. a reported revenue range by segment is not
guidance). Guidance is a number, range, or one-sided bound management states
for a quarter, half, fiscal year, or multi-year period that has not yet been
reported.

Every item must carry a `quote`: the exact sentence (or clause) from the
transcript it came from, copied verbatim, not paraphrased or summarized. If
you cannot point to a verbatim sentence supporting an item, do not include
that item.

If management explicitly declined to give guidance, or gave none at all
during the call, set `no_guidance_given` to true and return an empty `items`
list. Do not invent items to fill the list.

Map every guided period to the company's FISCAL period, not the calendar
quarter. Companies often run fiscal years that do not match the calendar
year (for example, NVIDIA's fiscal year ends in late January, so NVIDIA's
"Q4" guidance given in November refers to the quarter ending around January
of the following calendar year). Use context from the transcript (fiscal
year mentioned, "next quarter", "full year") to resolve this correctly
rather than assuming fiscal year equals calendar year.

Normalize every figure's unit into one of: usd, usd_thousands, usd_millions,
usd_billions, usd_per_share, percent, ratio. If management says "$2.5
billion to $2.7 billion", unit is usd_billions with low=2.5, high=2.7. A
one-sided bound ("at least $2 billion") sets only low or only high, leaving
the other null. A point estimate sets low and high to the same value.

Set `basis` to gaap or non_gaap when management says so explicitly;
otherwise unspecified.

Set `confidence` lower for vague or hedged language ("we're hopeful for
roughly...") and higher for a specific stated range.

Use `extraction_notes` only to flag anything ambiguous you resolved by
judgment call (an unclear fiscal year reference, a metric that could be
read two ways). Leave it null otherwise.
"""


def build_user_prompt(transcript: str) -> str:
    return (
        "Extract all forward-looking guidance from this earnings call "
        "transcript.\n\n---\n\n" + transcript
    )
