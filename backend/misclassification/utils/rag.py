from openai import OpenAI
from misclassification.utils.prompts import *
import json
import re


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
        """
        Translate text to the target language using the LLM.
        """
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
        """
        Determine if the text is a 'question' or a 'statement'.
        Return True if statement otherwise False.
        """
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
    Used to get scientific answers with inline source citations.
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
        Query ClimateGPT for a scientific answer.
        Explicitly asks for named sources to improve reference extraction.

        Returns:
            answer (str): Full scientific answer from ClimateGPT
            references (list): Source names extracted from the answer
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"{query} "
                            f"Please cite specific named sources such as IPCC, NOAA, "
                            f"NASA, or peer-reviewed journals in your answer."
                        ),
                    }
                ],
                extra_headers={"x-litellm-api-key": self.api_key},
            )
            answer = response.choices[0].message.content.strip()
            references = self._extract_references(answer)
            return answer, references

        except Exception as e:
            return "Climate science indicates this topic requires careful analysis.", []

    def _extract_references(self, text: str) -> list:
        """
        Extract source references mentioned in ClimateGPT's response.
        Only matches specific named organisations and publications.
        Never returns generic words like 'Science' or 'Nature' without context.
        """
        # Only match these specific known sources as whole words/phrases
        known_sources = [
            "IPCC", "NOAA", "NASA", "EPA", "WMO", "UNEP", "WHO",
            "World Bank", "IEA", "Met Office", "Copernicus",
            "Carbon Brief", "Nature Climate Change",
            "Environmental Research Letters",
            "National Oceanic and Atmospheric Administration",
            "Intergovernmental Panel on Climate Change",
            "National Aeronautics and Space Administration",
        ]

        found = []
        for source in known_sources:
            # Match as whole word/phrase, case-insensitive
            pattern = r'\b' + re.escape(source) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                # Use the short name for display
                display_name = source
                if source == "National Oceanic and Atmospheric Administration":
                    display_name = "NOAA"
                elif source == "Intergovernmental Panel on Climate Change":
                    display_name = "IPCC"
                elif source == "National Aeronautics and Space Administration":
                    display_name = "NASA"
                if display_name not in found:
                    found.append(display_name)

        # Only return if we found real named sources
        if found:
            return list(set(found))

        # Honest fallback — ClimateGPT itself is the source
        return ["ClimateGPT — Erasmus.AI (climategpt.ai)"]
