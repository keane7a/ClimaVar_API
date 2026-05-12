from openai import OpenAI
from misclassification.utils.prompts import *
import json


# Supported languages
SUPPORTED_LANGUAGES = ["english", "portuguese", "spanish"]


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
        Responds directly in the user's language.
        """
        system = (
            f"You are a climate science assistant with knowledge of IPCC reports "
            f"and scientific consensus. Give a brief, accurate answer in 2-3 sentences. "
            f"You MUST respond in {language}."
        )
        user = (
            f"Evaluate this climate claim or question based on climate science:\n\n"
            f"\"{query}\"\n\n"
            f"Respond in {language}."
        )
        return self.invoke(user, system=system, temperature=0.0)

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
        )
        return text_lower.endswith("?") or any(
            text_lower.startswith(w) for w in q_starters
        )


class ClimateGPTClient:
    """
    Client for the ClimateGPT API by Erasmus.AI.
    Receives original user input directly — no translation.
    Returns is_backup=True if ClimateGPT is unavailable.
    max_tokens=200 for faster responses while preserving verdict signal.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://models.erasmus.ai/v1",
        )
        self.model = "climategpt_8b_latest"

    def get_scientific_answer(self, query: str) -> tuple:
        """
        Query ClimateGPT with the original user input.
        No translation — ClimateGPT handles EN/PT/ES natively.
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
