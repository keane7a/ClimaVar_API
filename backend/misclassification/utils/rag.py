from openai import OpenAI
from misclassification.utils.prompts import *
import json
import re


# Supported languages
SUPPORTED_LANGUAGES = ["english", "portuguese", "spanish", "arabic"]

# Arabic script (basic, supplement and extended-A blocks)
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


def arabic_ratio(text: str) -> float:
    """Proportion of letters in the text that are Arabic (0.0 to 1.0)."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ARABIC_RE.match(c)) / len(letters)


def is_arabic(text: str) -> bool:
    """True if the text is predominantly written in Arabic script."""
    return arabic_ratio(text) >= 0.3


def language_label(language: str) -> str:
    """
    Name used inside prompts. Arabic is made explicit as Modern Standard
    Arabic so GPT-4o-mini does not mix dialects or switch to English.
    """
    if language and language.lower() == "arabic":
        return "Modern Standard Arabic"
    return language


class LLMClient:
    def __init__(self, client: OpenAI, model: str, temperature: float):
        self.client = client
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt: str, system: str = None, temperature=None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        res = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature if temperature is not None else self.temperature,
            messages=messages,
        )
        return res.choices[0].message.content.strip()

    def get_backup_answer(self, query: str, language: str) -> str:
        """
        Fallback scientific answer using GPT-4o-mini when ClimateGPT is offline.
        Responds directly in the user's language (Arabic included).
        """
        lang = language_label(language)
        system = (
            f"You are a climate science assistant with knowledge of IPCC reports "
            f"and scientific consensus. Give a brief, accurate answer in 2-3 sentences. "
            f"You MUST respond in {lang}."
        )
        user = (
            f"Evaluate this climate claim or question based on climate science:\n\n"
            f"\"{query}\"\n\n"
            f"Respond in {lang}."
        )
        return self.invoke(user, system=system, temperature=0.0)

    def translate_to_english(self, text: str) -> str:
        """
        Translate the user input to English.
        Used only for Arabic, before ClimateGPT and the classifiers,
        because ClimateGPT's Arabic support is not confirmed.
        """
        system = (
            "Translate the user's text into English exactly. "
            "Do not add or assume any context. Return only the translation."
        )
        return self.invoke(text, system=system, temperature=0.0)

    def ensure_language(self, text: str, language: str) -> str:
        """
        Final language check.
        If the input language is Arabic and less than half of the letters
        in the answer are Arabic, translate the whole answer to Arabic.
        Other languages are returned unchanged.
        """
        if not language or language.lower() != "arabic":
            return text
        if arabic_ratio(text) >= 0.5:
            return text

        system = (
            "You are a professional translator. Translate the text into Modern "
            "Standard Arabic, keeping the meaning, tone and the referee call at "
            "the start. Return only the translation."
        )
        return self.invoke(text, system=system, temperature=0.0)

    def is_climate_related(self, query: str) -> bool:
        """
        Check if the query is climate-related.
        Returns True if climate-related, False otherwise.
        """
        result = self.invoke(
            PROMPT_CLIMATE_TEXT_CLASSIFICATION.replace(
                "{user_question}", query
            ),
            temperature=0.0,
        )
        classify_clean = result.strip().lower()
        if (
            "output: 0" in classify_clean
            or classify_clean.endswith("\n0")
            or classify_clean.endswith(" 0")
            or classify_clean.strip() == "0"
        ):
            return False
        return True

    def is_question(self, query: str) -> bool:
        """
        Quick heuristic to detect if input is a question.
        Questions rarely need the sanity check since they are
        not claims that can be oversimplified.
        """
        text_lower = query.lower().strip()
        q_starters = (
            "what", "who", "where", "when", "why", "how",
            "does", "is ", "are ", "do ", "can ", "will ",
            "could ", "would ", "should ", "has ", "have ",
            # Portuguese
            "qual", "quais", "quando", "onde", "como", "por que",
            "é ", "são ", "existe", "pode", "será",
            # Spanish
            "qué", "cuál", "cuándo", "dónde", "cómo", "por qué",
            "es ", "son ", "existe", "puede", "será",
            # Arabic
            "هل", "ما ", "ماذا", "لماذا", "كيف", "متى", "أين", "أي ",
        )
        return (
            text_lower.endswith("?")
            or text_lower.endswith("؟")  # Arabic question mark
            or any(text_lower.startswith(w) for w in q_starters)
        )


class ClimateGPTClient:
    """
    Client for the ClimateGPT API by Erasmus.AI.
    Receives the user input directly (Arabic arrives already translated
    to English by views.py).
    Returns is_backup=True if ClimateGPT is unavailable.
    max_tokens=200 for faster responses while preserving verdict signal.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://models.climategpt.app/v1",
        )
        self.model = "climategpt4-moe36-int4"

    def get_scientific_answer(self, query: str) -> tuple:
        """
        Query ClimateGPT with the user input.
        EN/PT/ES are sent as original text; Arabic is sent in English.
        max_tokens=200 reduces latency while keeping enough context
        for accurate classification.

        Returns:
            answer (str): Scientific answer from ClimateGPT
            is_backup (bool): True if ClimateGPT was unavailable
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=200,
                messages=[{"role": "user", "content": query}],
                extra_headers={"x-litellm-api-key": self.api_key},
                timeout=15,
            )
            answer = response.choices[0].message.content.strip()
            return answer, False

        except Exception as e:
            return None, True
