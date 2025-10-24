from openai import OpenAI
from langdetect import detect
from pydantic import BaseModel
from typing import List
from utils.prompts import *


class LLMClient:
    def __init__(self, client: OpenAI, model: str, temperature: float):
        self.client = client
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt: str):
        """
        Call your general LLM (OpenAI) and return the message content.
        """
        res = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return res.choices[0].message.content.strip()

    def translate_language(self, text, target_lang):
        """
        Translate text to the target language using the LLM.
        Args:
            text (str): The text to be translated.
            target_lang (str): The target language for translation. Choices: "en": English, "pt": Portuguese, "es": Spanish
        Returns:
            tuple: (detected_language, translated_text) if translation is needed, else (detected_language, original_text)
        """

        if target_lang not in ["en", "pt", "es"]:
            return False

        lang_type = detect(text)
        # check language type
        if lang_type == target_lang:
            return lang_type, text

        lang_map = {"pt": "Portuguese", "en": "English", "es": "Spanish"}
        prompt = f"Translate the following text from {lang_map[lang_type]} to {lang_map[target_lang]}:\n\n{text}"
        translation = self.invoke(prompt)
        return lang_type, translation

    def get_text_type(self, text: str):
        """
        Determine if the text is a 'question' or a 'statement'.
        Return True if statement otherwise False.
        """

        # Simple heuristic check first for speed
        text_lower = text.lower().strip()
        q_words = (
            "what",
            "who",
            "where",
            "when",
            "why",
            "how",
            "does",
            "is ",
            "are ",
            "do ",
        )
        if text_lower.endswith("?") or text.startswith(q_words):
            return False

        # Fallback to LLM for ambiguity
        prompt = (
            "Classify the following text as either 'Question' or 'Statement'. "
            "Respond with only one word: Question or Statement.\n\n"
            f"Text: {text}\n"
            "Classification:"
        )
        # Call the LLM to get the classification
        text_type = self.invoke(prompt).strip().lower()

        if "question" in text_type:
            return False

        return True


# Data models for CARDS response
class Category(BaseModel):
    category_number: str
    category_name: str
    justification: str  # Optional: remove this field if you don't want to include it in the response


class Categories(BaseModel):
    categories: List[Category]


class CARDSClient:
    def __init__(self, client: OpenAI, model: str, temperature: float):
        self.client = client
        self.model = model
        self.temperature = temperature

        pass

    def classify_claim(self, text):
        """
        Classify a claim with CARDS model.

        Parameters
        ----------
        text : str
        The user's claim/question.

        Returns
        -------
        is_misinformation : bool
            True if the text is classified as misinformation, False otherwise.
        categories : List[Category]
        """

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "user", "content": text},
            ],
            response_format=Categories,
            extra_body={"prompt_id": "cards"},
            temperature=self.temperature,
        )
        parsed = response.choices[0].message.parsed
        category_number = parsed.categories[0].category_number

        # Parse categories
        bits = [
            f"{c.category_number}: {c.category_name}, justification: {c.justification}"
            for c in parsed.categories
        ]
        categories = "Categories=" + "; ".join(bits)

        # Text is has no misinformation
        if len(parsed.categories) == 1 and category_number == "0_0_0":
            return False, categories

        # text is misinformation
        return True, categories
