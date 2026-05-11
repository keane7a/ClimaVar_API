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
        """
        Call OpenAI and return the message content.
        Supports optional system message for better instruction-following.
        """
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
            f'"{query}"\n\n'
            f"Respond in {language}."
        )
        return self.invoke(user, system=system, temperature=0.0)


class ClimateGPTClient:
    """
    Client for the ClimateGPT API by Erasmus.AI.
    Receives original user input directly — no translation.
    Returns is_backup=True if ClimateGPT is unavailable.
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

        Returns:
            answer (str): Scientific answer from ClimateGPT
            is_backup (bool): True if ClimateGPT was unavailable
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": query}],
                extra_headers={"x-litellm-api-key": self.api_key},
                timeout=15,
            )
            answer = response.choices[0].message.content.strip()
            return answer, False

        except Exception as e:
            return None, True
