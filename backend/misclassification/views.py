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
        """
        Quick heuristic to detect English text.
        Saves 1-2 seconds by skipping translation for English queries.
        Only uses words that are uniquely English and do not appear
        in Portuguese or Spanish — avoids false positives on multilingual
        words like 'global', 'climate', 'solar', 'natural'.
        """
        if not text:
            return True

        text_lower = text.lower().strip()
        words = text_lower.split()

        # Words that are uniquely English and rarely appear in Portuguese/Spanish
        english_only_words = [
            'is', 'are', 'does', 'do', 'can', 'will', 'what', 'how',
            'why', 'when', 'where', 'the', 'this', 'that', 'have',
            'has', 'been', 'their', 'there', 'because', 'which', 'would',
            'warming', 'weather', 'carbon', 'emissions',
        ]

        has_english_words = any(word in words for word in english_only_words)

        non_ascii = sum(1 for char in text if ord(char) > 127)
        mostly_ascii = (non_ascii / len(text)) < 0.15

        # Both conditions must be true
        return has_english_words and mostly_ascii

    def _classify_claim(self, scientific_answer: str) -> int:
        """
        Ask GPT-4o-mini to classify the claim into one of three categories
        based on ClimateGPT's scientific answer:

        0 = ACCURATE — claim is fully supported by climate science
        1 = MISINFORMATION — claim is false or clearly misleading
        2 = PARTIAL — claim has some truth but is oversimplified,
                      exaggerated, missing important context,
                      or only partially correct

        Returns integer: 0, 1, or 2
        """
        system = (
            "You are a climate fact-checking assistant. "
            "You must reply with exactly ONE digit: 0, 1, or 2. No other text."
        )
        user = (
            f"Read this climate science response carefully:\n\n"
            f"{scientific_answer}\n\n"
            f"Based on this response, classify the original statement:\n\n"
            f"Reply with exactly one digit:\n"
            f"0 = The statement is ACCURATE and fully supported by climate science\n"
            f"1 = The statement is FALSE or clearly MISLEADING misinformation\n"
            f"2 = The statement is PARTIALLY TRUE but oversimplified, exaggerated, "
            f"or missing important context that changes its meaning\n\n"
            f"Reply with only 0, 1, or 2."
        )
        result = llm_client.invoke(
            user, system=system, temperature=0.0
        ).strip()

        # Extract first digit found
        for char in result:
            if char in ("0", "1", "2"):
                return int(char)

        # Default to accurate if unclear
        return ACCURATE

    def _build_football_answer(
        self,
        original_query: str,
        scientific_answer: str,
        verdict: int,
    ) -> str:
        """
        Generates a football-style ClimaVAR verdict using GPT-4o-mini.

        verdict=0 (ACCURATE)       → GOAL! / PLAY ON! / VAR CONFIRMS! / FAIR PLAY!
        verdict=1 (MISINFORMATION) → RED CARD! / OFFSIDE! / FOUL!
        verdict=2 (PARTIAL)        → YELLOW CARD! only
        """
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
        ClimateGPT pipeline:

        1.  Cache check
        2.  Validate length
        3.  Smart English detection — only uses uniquely English words
        4.  Climate relevance check — robust Output: 0/1 parsing
        5.  Detect question vs statement
        6.  Build context-aware ClimateGPT query
        7.  ClimateGPT scientific answer
        8.  Classify: 0=accurate, 1=misinfo, 2=partial
            (questions are always 0)
        9.  Generate football verdict
        10. Translate back if needed
        11. Log (is_misinformation: 0, 1, or 2)
        12. Cache + return
        """

        query = request.data.get("text", "")

        # 1) Cache check
        cache_key = f"climavar_query_{hashlib.md5(query.lower().encode()).hexdigest()}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return Response(cached_result, status=status.HTTP_200_OK)

        # 2) Validate length
        if not 10 < len(query) < 300:
            return Response(
                {"message": "Input text must be between 10 and 300 characters."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 3) Smart translation — skip only if confidently English
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

        # 4) Climate relevance check — robust output parsing
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

        # 6) Build context-aware ClimateGPT query
        # Questions sent as-is so ClimateGPT answers directly.
        # Statements framed as accuracy checks so ClimateGPT evaluates them.
        if not is_statement:
            climategpt_query = query_english
        else:
            climategpt_query = (
                f"Is the following statement accurate based on climate science? "
                f"{query_english}"
            )

        # 7) Get scientific answer from ClimateGPT
        scientific_answer, references = climategpt_client.get_scientific_answer(
            climategpt_query
        )

        # 8) Classify verdict — questions are always ACCURATE (0)
        if not is_statement:
            verdict = ACCURATE
        else:
            verdict = self._classify_claim(scientific_answer)

        # 9) Generate football-style answer
        llm_answer = self._build_football_answer(
            original_query=query_english,
            scientific_answer=scientific_answer,
            verdict=verdict,
        )

        # 10) Translate back if needed
        if src_lang != "English":
            final_answer = llm_client.translate_language(llm_answer, src_lang)[1]
        else:
            final_answer = llm_answer

        # 11) Log — is_misinformation is 0, 1, or 2
        MisclassificationLog.objects.create(
            user=request.user,
            user_input=request.data.get("text", ""),
            llm_output=final_answer,
            is_misinformation=verdict,
            references="\n".join(references),
        )

        # 12) Cache + return
        result = {
            "llm_response": final_answer,
            "misinformation": verdict,
            "references": references,
        }
        cache.set(cache_key, result, timeout=86400)

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
