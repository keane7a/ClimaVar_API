from http.client import HTTPException
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.permissions import IsAdminUser, AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import action
from drf_spectacular.utils import extend_schema_view, extend_schema
from misclassification.models import MisclassificationLog
from misclassification.serializer import MisclassificationLogSerializer
from utils.docs_utils import (
    CHECK_MISCLASSIFICATION_RESPONSES,
    CHECK_MISCLASSIFICATION_EXAMPLES,
    CHECK_MISCLASSIFICATION_REQUEST,
)
from permissions import isAdminAndReadListOnly

from openai import OpenAI
from misclassification.utils.rag import LLMClient, ClimateGPTClient
from misclassification.utils.prompts import *
import os
import re
from django.core.cache import cache
import hashlib


# Verdict constants
ACCURATE = 0
MISINFORMATION = 1
PARTIAL = 2

# Init GPT-4o-mini — translation, climate check, football answer generation
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm_client = LLMClient(openai_client, model="gpt-4o-mini", temperature=0.2)

# Init ClimateGPT — scientific answer retrieval
climategpt_client = ClimateGPTClient(api_key=os.getenv("CLIMATEGPT_API_KEY"))


class MisclassificationViewSet(viewsets.ViewSet):
    permission_classes = [IsAdminUser]

    def _is_likely_english(self, text):
        if not text:
            return True
        non_english_chars = set('àáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿãõç')
        if any(char in non_english_chars for char in text.lower()):
            return False
        text_lower = text.lower().strip()
        words = set(text_lower.split())
        english_only_words = {
            'the', 'this', 'that', 'these', 'those',
            'have', 'has', 'been', 'being',
            'their', 'there', 'they', 'them',
            'because', 'which', 'would', 'could', 'should',
            'warming', 'weather', 'rainfall', 'flooding',
            'aren', 'isn', 'doesn', 'didn', 'wasn', 'weren',
        }
        if words & english_only_words:
            return True
        return False

    def _classify_claim(self, scientific_answer: str) -> int:
        """
        Classify based on ClimateGPT's answer.
        Reads TRUE/FALSE/PARTIALLY TRUE signal from first sentence.
        Falls back to GPT-4o-mini if signal is unclear.
        """
        first_sentence = scientific_answer.split('.')[0].strip().upper()

        if any(word in first_sentence for word in [
            'FALSE', 'INCORRECT', 'INACCURATE', 'NOT TRUE', 'WRONG',
            'MISLEADING', 'NOT ACCURATE', 'NOT SUPPORTED',
        ]):
            return MISINFORMATION

        if any(word in first_sentence for word in [
            'PARTIALLY TRUE', 'PARTIALLY CORRECT', 'PARTLY TRUE',
            'PARTIALLY ACCURATE', 'MIXED', 'NUANCED',
        ]):
            return PARTIAL

        if any(word in first_sentence for word in [
            'TRUE', 'CORRECT', 'ACCURATE', 'SUPPORTED', 'CONFIRMED',
            'YES', 'INDEED',
        ]):
            return ACCURATE

        # Fallback classifier
        system = (
            "You are a climate fact-checking assistant. "
            "Reply with exactly ONE digit: 0, 1, or 2. No other text."
        )
        user = (
            f"Read the first sentence of this climate science response:\n\n"
            f'"{first_sentence}"\n\n'
            f"Does this indicate the original statement is:\n"
            f"0 = TRUE and accurate\n"
            f"1 = FALSE or misinformation\n"
            f"2 = PARTIALLY TRUE but needs context\n\n"
            f"Reply with only 0, 1, or 2."
        )
        result = llm_client.invoke(
            user, system=system, temperature=0.0
        ).strip()

        for char in result:
            if char in ("0", "1", "2"):
                return int(char)

        return ACCURATE

    def _classify_backup(self, query: str) -> int:
        """
        Classify directly using GPT-4o-mini when ClimateGPT is offline.
        Uses its own knowledge of scientific consensus.
        """
        system = (
            "You are a climate science fact-checker with expert knowledge "
            "of IPCC reports and scientific consensus. "
            "Reply with exactly ONE digit: 0, 1, or 2. No other text."
        )
        user = (
            f"Classify this climate statement based on scientific consensus:\n\n"
            f'"{query}"\n\n'
            f"0 = ACCURATE: Correct and supported by scientific consensus.\n"
            f"1 = MISINFORMATION: Clearly false or contradicts scientific consensus.\n"
            f"2 = PARTIAL: Partially true but misleading or oversimplified.\n\n"
            f"EXAMPLES:\n"
            f"'Greenhouse gas concentrations reached record levels in 2023' → 0\n"
            f"'Human activities are the primary cause of climate change' → 0\n"
            f"'Climate change is not caused by human activity' → 1\n"
            f"'Global warming is a hoax invented by scientists' → 1\n"
            f"'Electric vehicles produce zero emissions' → 2\n"
            f"'Planting trees alone can solve climate change' → 2\n\n"
            f"Reply with only 0, 1, or 2."
        )
        result = llm_client.invoke(
            user, system=system, temperature=0.0
        ).strip()

        for char in result:
            if char in ("0", "1", "2"):
                return int(char)

        return ACCURATE

    def _build_football_answer(
        self,
        original_query: str,
        scientific_answer: str,
        verdict: int,
    ) -> str:
        if verdict == MISINFORMATION:
            system = """You are ClimaVAR, a climate fact-checker that speaks like a football VAR referee.

ABSOLUTE RULES:
- Your entire response must be ONE sentence, maximum 280 characters.
- You MUST start with exactly ONE of these calls: RED CARD! or OFFSIDE! or FOUL!
- Never use GOAL!, PLAY ON!, VAR CONFIRMS!, FAIR PLAY!, or YELLOW CARD! for false claims.
- Never combine two calls. Pick ONE only.
- No preamble. No explanation. Just the single sentence starting with the call."""

            user = (
                f'The user said: "{original_query}"\n\n'
                f"A climate scientist found this is FALSE or MISLEADING.\n\n"
                f"Scientific explanation:\n{scientific_answer}\n\n"
                f"Write ONE sentence (max 280 chars) starting with "
                f"RED CARD! or OFFSIDE! or FOUL! "
                f"that refutes the claim using the evidence above."
            )

        elif verdict == PARTIAL:
            system = """You are ClimaVAR, a climate fact-checker that speaks like a football VAR referee.

ABSOLUTE RULES:
- Your entire response must be ONE sentence, maximum 280 characters.
- You MUST start with exactly: YELLOW CARD!
- This is for claims that are partially true but oversimplified or missing crucial context.
- Your tone should be cautionary — "not the full picture", "needs context", "take care".
- Never use RED CARD!, OFFSIDE!, FOUL!, GOAL!, PLAY ON!, VAR CONFIRMS!, FAIR PLAY!
- No preamble. Just the single sentence starting with YELLOW CARD!"""

            user = (
                f'The user said: "{original_query}"\n\n'
                f"A climate scientist found this is PARTIALLY TRUE but incomplete "
                f"or oversimplified.\n\n"
                f"Scientific explanation:\n{scientific_answer}\n\n"
                f"Write ONE sentence (max 280 chars) starting with YELLOW CARD! "
                f"that warns the user this needs more context, "
                f"using the evidence above."
            )

        else:  # ACCURATE
            system = """You are ClimaVAR, a climate fact-checker that speaks like a football VAR referee.

ABSOLUTE RULES:
- Your entire response must be ONE sentence, maximum 280 characters.
- You MUST start with exactly ONE of these calls: GOAL! or PLAY ON! or VAR CONFIRMS! or FAIR PLAY!
- Never use RED CARD!, OFFSIDE!, FOUL!, YELLOW CARD! for accurate claims.
- Never combine two calls together.
- No preamble. Just the single sentence starting with the call."""

            user = (
                f'The user said: "{original_query}"\n\n'
                f"A climate scientist confirmed this is ACCURATE.\n\n"
                f"Scientific explanation:\n{scientific_answer}\n\n"
                f"Write ONE sentence (max 280 chars) starting with "
                f"GOAL! or PLAY ON! or VAR CONFIRMS! or FAIR PLAY! "
                f"that confirms the claim using the evidence above."
            )

        return llm_client.invoke(user, system=system, temperature=0.0)

    @extend_schema(
        summary="Check Misclassification",
        description="Evaluate a user-supplied climate statement and return an LLM response in a football like manner.",
        request=CHECK_MISCLASSIFICATION_REQUEST,
        responses=CHECK_MISCLASSIFICATION_RESPONSES,
        examples=CHECK_MISCLASSIFICATION_EXAMPLES,
    )
    @action(
        detail=False,
        methods=["post"],
        permission_classes=[IsAuthenticated],
        url_path="check-misclassification",
    )
    def check_misclassification(self, request):
        """
        ClimateGPT pipeline with backup:

        1.  Cache check — DISABLED FOR TESTING
        2.  Validate length
        3.  Smart English detection
        4.  Climate relevance check
        5.  Detect question vs statement
        6.  Try ClimateGPT — if offline, use GPT-4o-mini backup
        7.  Classify verdict
        8.  Generate football verdict
        9.  Translate back if needed
        10. Log
        11. Cache — DISABLED FOR TESTING
        """

        query = request.data.get("text", "")

        # 1) Cache — DISABLED FOR TESTING
        cache_key = f"climavar_query_{hashlib.md5(query.lower().encode()).hexdigest()}"
        # cached_result = cache.get(cache_key)
        # if cached_result:
        #     return Response(cached_result, status=status.HTTP_200_OK)

        # 2) Validate length
        if not 10 < len(query) < 300:
            return Response(
                {"message": "Input text must be between 10 and 300 characters."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 3) Smart translation
        if self._is_likely_english(query):
            src_lang = "English"
            query_english = query
        else:
            src_lang, query_english = llm_client.translate_language(
                query, "English"
            )
            if not (query_english and src_lang):
                return Response(
                    {"message": "Unsupported language for translation."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        # 4) Climate relevance check
        classify_claim = llm_client.invoke(
            PROMPT_CLIMATE_TEXT_CLASSIFICATION.replace(
                "{user_question}", query_english
            ),
            temperature=0.0,
        )
        classify_clean = classify_claim.strip().lower()
        if (
            "output: 0" in classify_clean
            or classify_clean.endswith("\n0")
            or classify_clean.endswith(" 0")
            or classify_clean.strip() == "0"
        ):
            return Response(
                {
                    "message": (
                        "Query does not contain climate-related topics or sufficient "
                        "climate-related keywords. Please rephrase your query to focus "
                        "on climate-related content."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 5) Detect question vs statement
        is_statement = llm_client.get_text_type(query_english)

        # 6) Try ClimateGPT — fallback to GPT-4o-mini if offline
        scientific_answer, references, is_backup = (
            climategpt_client.get_scientific_answer(
                query_english,
                is_statement=is_statement,
            )
        )

        if is_backup:
            # ClimateGPT is offline — use GPT-4o-mini backup
            scientific_answer, references = llm_client.get_backup_answer(
                query_english, is_statement
            )

        # 7) Classify verdict — questions always ACCURATE (0)
        if not is_statement:
            verdict = ACCURATE
        elif is_backup:
            # Use direct classification when ClimateGPT is offline
            verdict = self._classify_backup(query_english)
        else:
            # Use ClimateGPT's answer to classify
            verdict = self._classify_claim(scientific_answer)

        # 8) Generate football-style answer
        llm_answer = self._build_football_answer(
            original_query=query_english,
            scientific_answer=scientific_answer,
            verdict=verdict,
        )

        # 9) Translate back if needed
        if src_lang != "English":
            final_answer = llm_client.translate_language(llm_answer, src_lang)[1]
        else:
            final_answer = llm_answer

        # 10) Log
        MisclassificationLog.objects.create(
            user=request.user,
            user_input=request.data.get("text", ""),
            llm_output=final_answer,
            is_misinformation=verdict,
            references="\n".join(references),
        )

        # 11) Cache — DISABLED FOR TESTING
        result = {
            "llm_response": final_answer,
            "misinformation": verdict,
            "references": references,
            "source": "backup" if is_backup else "climategpt",
        }
        # cache.set(cache_key, result, timeout=86400)

        return Response(result, status=status.HTTP_200_OK)


@extend_schema_view(
    list=extend_schema(
        summary="Get Misclassification Logs",
        description="Retrieve a list of all misclassification logs.",
    ),
    retrieve=extend_schema(exclude=True),
    create=extend_schema(exclude=True),
    update=extend_schema(exclude=True),
    partial_update=extend_schema(exclude=True),
    destroy=extend_schema(exclude=True),
)
class MisclassificationLogViewSet(viewsets.ModelViewSet):
    queryset = MisclassificationLog.objects.all()
    serializer_class = MisclassificationLogSerializer
    permission_classes = [IsAdminUser]
