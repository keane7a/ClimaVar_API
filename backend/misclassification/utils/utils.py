import re
from typing import Dict, Tuple
import unicodedata

PATTERNS = {
    "science": [
        r"\bclimate\b",
        r"\bglobal\s+warming\b",
        r"\bmitigat(?:e|ion)\b",
        r"\badaptat(?:e|ion)\b",
        r"\banthropogen(?:ic|y)\b",
        r"\bemission(?:s)?\b",
        r"\bnet[\s-]?zero\b",
        r"\bcarbon\s+budget\b",
        r"\bradiative\s+forcing\b",
        r"\btipping\s+point(?:s)?\b",
    ],
    "impacts": [
        r"\bdrought(?:s)?\b",
        r"\bheat[\s-]?wave(?:s)?\b",
        r"\bflood(?:s|ing)?\b",
        r"\bsea[\s-]?level\b",
        r"\bglacier(?:s)?\b",
        r"\bpermafrost\b",
        r"\bwildfire(?:s)?\b",
        r"\bocean\s+acidification\b",
        r"\bprecip(?:itation)?\b",
    ],
    "gases": [
        r"\bco2\b",
        r"\bcarbon\s+dioxide\b",
        r"\bmethane\b",
        r"\bghg(?:s)?\b",
        r"\bgreenhouse\b",
        r"\baerosol(?:s)?\b",
        r"\bn2o\b",
    ],
    "institutions": [
        r"\bipcc\b",
        r"\bunfccc\b",
        r"\bunep\b",
        r"\bwmo\b",
        r"\bnasa\b",
        r"\bnoaa\b",
        r"\biea\b",
        r"\bar6\b",
        r"\bsynthesis\s+report\b",
        r"\bnca\b",
    ],
    "sectors": [
        r"\brenewable(?:s)?\b",
        r"\bwind\b",
        r"\bsolar\b",
        r"\belectric\s+vehicle(?:s)?\b",
        r"\bev\b",
        r"\bdecarboni[sz]e\b",
        r"\bcarbon\s+capture\b",
        r"\bccs\b",
    ],
}
COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in PATTERNS.items()}


def climate_keyword_score(text_en: str) -> Tuple[bool, Dict[str, int]]:
    """
    Returns (is_climate, counts_per_bucket) by scoring keyword hits.
    Rule: (2*science + 2*impacts + gases + institutions + sectors) >= 2 AND (science>0 OR impacts>0)
    """
    t = (
        unicodedata.normalize("NFKD", text_en)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .strip()
    )
    counts = {k: 0 for k in COMPILED}
    for bucket, regs in COMPILED.items():
        for rgx in regs:
            if rgx.search(t):
                counts[bucket] += 1
    score = (
        2 * counts["science"]
        + 2 * counts["impacts"]
        + counts["gases"]
        + counts["institutions"]
        + counts["sectors"]
    )
    is_climate = (score >= 2) and (counts["science"] > 0 or counts["impacts"] > 0)
    return is_climate
