from openai import OpenAI
from misclassification.utils.prompts import *
import json


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

    def translate_language(self, text: str, target_lang: str):
        if target_lang.lower() not in ["english", "portuguese", "spanish"]:
            return None, None

        if not text or not text.strip():
            return None, None

        prompt = f"""
        You are a multilingual assistant. 
        Detect the language of the following text and translate it into {target_lang} if needed. You should translate it EXACTLY and DO NOT assume any context beyond the text provided.
        ONLY English, Spanish, and Portuguese are supported otherwise detected_language = "".
        If the text is already in {target_lang}, do not translate it again and translation = user's text.
        
        Return the result in JSON format exactly like this:
        {{
            "detected_language": "<detected_language>",
            "translation": "<translated_text>"
        }}

        Text:
        {text}
        """

        response = self.invoke(prompt)
        data = json.loads(response)
        detected_lang = data.get("detected_language", None)
        translation = data.get("translation", None)
        return detected_lang, translation

    def get_text_type(self, text: str):
        text_lower = text.lower().strip()
        q_words = (
            "what", "who", "where", "when", "why", "how",
            "does", "is ", "are ", "do ",
        )
        if text_lower.endswith("?") or text.startswith(q_words):
            return False

        prompt = (
            "Classify the following text as either 'Question' or 'Statement'. "
            "Respond with only one word: Question or Statement.\n\n"
            f"Text: {text}\n"
            "Classification:"
        )
        text_type = self.invoke(prompt).strip().lower()

        if "question" in text_type:
            return False

        return True


class ClimateGPTClient:
    """
    Client for the ClimateGPT API by Erasmus.AI.
    OpenAI-compatible endpoint with climate-specific knowledge base.
    Used to get scientific answers to ground the football-style verdict.
    References removed — not shown in the interface.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://models.erasmus.ai/v1",
        )
        self.model = "climategpt_8b_latest"

    def get_scientific_answer(self, query: str, is_statement: bool = False) -> tuple:
        """
        Query ClimateGPT for a scientific answer.

        For statements, instructs ClimateGPT to begin with a clear
        TRUE, FALSE, or PARTIALLY TRUE verdict before elaborating.
        This prevents academic hedging from confusing the classifier.

        Returns:
            answer (str): Full scientific answer from ClimateGPT
            references (list): Always empty — references not shown in UI
        """
        try:
            if is_statement:
                content = (
                    f"{query} "
                    f"Begin your answer by stating clearly whether this is "
                    f"TRUE, FALSE, or PARTIALLY TRUE according to climate science. "
                    f"Then explain why."
                )
            else:
                content = query

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                extra_headers={"x-litellm-api-key": self.api_key},
            )
            answer = response.choices[0].message.content.strip()
            return answer, []

        except Exception as e:
            return "Climate science indicates this topic requires careful analysis.", []
