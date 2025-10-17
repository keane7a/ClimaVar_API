import os
import re
import csv
import time
import hashlib
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv


from openai import OpenAI
from pydantic import BaseModel
from misclassification.utils.prompts import get_prompt


CARDS_BASE_URL = "https://api.discourselab.ai/v1"
CARDS_MODEL = "cards-mini-sonnet-2024-12-05"
CARDS_PROMPT_ID = "cards"   # required by API

load_dotenv()  # take environment variables from .env
CARDS_API_KEY = os.environ.get("CARDS_API_KEY")  # set: %env CARDS_API_KEY=...
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # set: %env OPENAI_API_KEY=...

if not CARDS_API_KEY:
    raise RuntimeError("Missing CARDS_API_KEY (set with %env CARDS_API_KEY=...)")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY (set with %env OPENAI_API_KEY=...)")

# Clients:
cards_client = OpenAI(api_key=CARDS_API_KEY, base_url=CARDS_BASE_URL)   # DiscourseLab CARDS
llm_client   = OpenAI(api_key=OPENAI_API_KEY)                           # OpenAI default


# Input Guard - paragraph limit
MAX_INPUT_PARAGRAPHS = 5
def within_paragraph_limit(text: str, max_paragraphs: int = MAX_INPUT_PARAGRAPHS) -> bool:
    """
    Return True if the message contains ≤ max_paragraphs non-empty paragraphs.
    """
    paras = [p for p in text.strip().split("\n") if p.strip()]
    return len(paras) <= max_paragraphs

def validate_input(user_text: str) -> Optional[str]:
    """
    Validate user input length only (scope/safety handled in prompts).
    """
    if not within_paragraph_limit(user_text):
        return f"Please shorten your message (max {MAX_INPUT_PARAGRAPHS} paragraphs)."
    return None


# CARDS MODELS AND CLASSIFICATION
class Category(BaseModel):
    """One CARDS category (number + human-readable name)."""
    category_number: str
    category_name: str

class Categories(BaseModel):
    """Top-level parsed payload holding a list of categories."""
    categories: List[Category]

@dataclass
class CardsResult:
    """Container for CARDS classification results used downstream."""
    is_misinformation: bool      # True if categories non-empty
    label: str                   # 'FALSE' if misinfo else 'TRUE'
    categories: List[Category]   # Pydantic models

def cards_classify_claim(claim_text: str) -> CardsResult:
    """
    Classify a user claim with CARDS.
    .parse(..., response_format=Categories, extra_body={'prompt_id':'cards'})
    """
    resp = cards_client.beta.chat.completions.parse(
        model=CARDS_MODEL,
        messages=[{"role": "user", "content": claim_text}],
        response_format=Categories,
        extra_body={"prompt_id": CARDS_PROMPT_ID},
        temperature=0
    )
    parsed: Categories = resp.choices[0].message.parsed
    cats = parsed.categories
    is_misinfo = len(cats) > 0
    label = "FALSE" if is_misinfo else "TRUE"
    return CardsResult(is_misinformation=is_misinfo, label=label, categories=cats)


# # Logging Helpers
# def prompt_hash(text: str) -> str:
#     """Stable 8-char hash of a prompt template for audit trail."""
#     return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

# @dataclass
# class RunCfg:
#     """
#     Configuration for a single generation run.
#     """
#     prompt_version: str = "s2"   # choose among: i0,i1,i2,s0,s1,s2
#     model: str = "gpt-4o-mini"
#     temperature: float = 0.2

# LOG_PATH = "runs_log.csv"

# def log_run(
#     user_text: str,
#     cards_label: str,
#     cards_categories: List[Category],
#     cfg: RunCfg,
#     prompt_family: str,
#     uses_cot: bool,
#     final_text: str
# ) -> None:
#     """
#     Append a row to runs_log.csv with rich metadata so your supervisor can compare:
#     - prompt_version, prompt_family, uses_chain_of_thought
#     - model, temperature
#     - CARDS label/categories
#     - user_text, final_text, length
#     """
#     exists = os.path.exists(LOG_PATH)
#     with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         if not exists:
#             w.writerow([
#                 "ts","prompt_version","prompt_hash","prompt_family","uses_chain_of_thought",
#                 "model","temperature",
#                 "cards_label","cards_categories",
#                 "user_text","final_chars","final_text"
#             ])
#         w.writerow([
#             int(time.time()),
#             cfg.prompt_version,
#             prompt_hash(get_prompt(cfg.prompt_version)["template"]),
#             prompt_family,
#             uses_cot,
#             cfg.model,
#             cfg.temperature,
#             cards_label,
#             " | ".join(f"{c.category_number}:{c.category_name}" for c in cards_categories) if cards_categories else "",
#             user_text,
#             len(final_text),
#             final_text
#         ])

# PROMPT BUILDER
def categories_summary(cards: CardsResult, max_items: int = 2) -> str:
    """
    Create a short, readable categories string for the prompt context.
    """
    if not cards.categories:
        return "Categories=None"
    bits = [f"{c.category_number}: {c.category_name}" for c in cards.categories[:max_items]]
    return "Categories=" + "; ".join(bits)

def build_prompt(user_question: str, cards: CardsResult, prompt_version: str):
    """
    Resolve template by version and render with:
    - user question
    - TRUE/FALSE label from CARDS
    - compact categories summary
    Also returns metadata (family, uses_cot) for logging.
    """
    spec = get_prompt(prompt_version)
    template = spec["template"]
    tf = "FALSE" if cards.is_misinformation else "TRUE"
    cat_sum = categories_summary(cards)
    prompt = (template
              .replace("{user_question}", user_question)
              .replace("{true_false}", tf)
              .replace("{categories_summary}", cat_sum))
    return prompt, spec["prompt_family"], spec["uses_chain_of_thought"]


# LLM CALLER + EXTRACTION
def llm_answer(prompt: str, model: str, temperature: float) -> str:
    """
    Call the OpenAI chat completion endpoint with the given settings.
    Returns raw assistant message content.
    """
    resp = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def extract_final(text: str) -> str:
    """
    Extract the 'FINAL:' line from the model output.
    If absent, fall back to the first line.
    Always trims to ≤300 chars.
    """
    for line in text.splitlines():
        if line.strip().lower().startswith("final:"):
            return line.split(":", 1)[1].strip()[:300]
    return text.strip().split("\n", 1)[0][:300]

