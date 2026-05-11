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
from misclassification.utils.rag import LLMClient, ClimateGPTClient, SUPPORTED_LANGUAGES
from misclassification.utils.prompts import *
import os
from django.core.cache import cache
import hashlib


# Verdict constants
ACCURATE = 0
MISINFORMATION = 1
PARTIAL = 2

# Language display names for messages
LANGUAGE_NAMES = {
    "english": "English",
    "portuguese": "Portuguese",
    "spanish": "Spanish",
}

# Init GPT-4o-mini
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm_client = LLMClient(openai_client, model="gpt-4o-mini", temperature=0.2)

# Init ClimateGPT
climategpt_client = ClimateGPTClient(api_key=os.getenv("CLIMATEGPT_API_KEY"))


class MisclassificationViewSet(viewsets.ViewSet):
    permission_classes = [IsAdminUser]

    def _classify_claim(self, scientific_answer: str) -> int:
        """
        Classify verdict by reading ClimateGPT's natural language patterns.
        Works across English, Portuguese and Spanish without translation.

        Looks for explicit FALSE/TRUE/PARTIAL signals in the first sentence.
        Falls back to GPT-4o-mini if signal is unclear.
        """
        first_sentence = scientific_answer.split('.')[0].strip().upper()

        # Explicit FALSE signals across EN/PT/ES
        false_signals = [
            'INCORRECT', 'MISLEADING', 'IS A MYTH', 'NO CREDIBLE EVIDENCE',
            'NOT SUPPORTED', 'IS FALSE', 'ASSERTION IS', 'STATEMENT IS INCORRECT',
            'THIS IS NOT', 'CLAIM IS FALSE', 'NOT TRUE', 'IS WRONG',
            # Portuguese
            'É INCORRETA', 'É INCORRETO', 'NÃO TÊM NADA', 'NÃO É VERDADE',
            'AFIRMAÇÃO É FALSA', 'CRENÇA É INCORRETA',
            # Spanish
            'ES INCORRECTA', 'ES INCORRECTA', 'ES FALSA', 'NO ES CIERTO',
            'LA AFIRMACIÓN ES', 'ES UN MITO',
        ]

        # Explicit TRUE signals across EN/PT/ES
        true_signals = [
            'SIGNIFICANT MILESTONE', 'UNPRECEDENTED', 'INDEED CONFIRMED',
            'OVERWHELMING CONSENSUS', 'WELL DOCUMENTED', 'SCIENTIFIC CONSENSUS',
            'IS ACCURATE', 'IS CORRECT', 'IS TRUE', 'HAS BEEN CONFIRMED',
            # Portuguese
            'CONSENSO CIENTÍFICO', 'É VERDADE', 'ESTÁ CORRETO', 'CONFIRMADO',
            'TEM UMA FORTE RELAÇÃO',
            # Spanish
            'CONSENSO CIENTÍFICO', 'ES VERDAD', 'ESTÁ CONFIRMADO', 'ES CORRECTO',
        ]

        # Explicit PARTIAL signals
        partial_signals = [
            'PARTIALLY CORRECT', 'PARTIALLY TRUE', 'PARTIALLY ACCURATE',
            'WHILE IT IS TRUE', 'WHILE SOME',
            # Portuguese
            'PARCIALMENTE', 'EMBORA SEJA VERDADE',
            # Spanish
            'PARCIALMENTE', 'AUNQUE ES VERDAD',
        ]

        if any(signal in first_sentence for signal in false_signals):
            return MISINFORMATION

        if any(signal in first_sentence for signal in partial_signals):
            return PARTIAL

        if any(signal in first_sentence for signal in true_signals):
            return ACCURATE

        # Fallback — ask GPT-4o-mini to read the signal
        system = (
            "You are a climate fact-checking assistant. "
            "Reply with exactly ONE digit: 0, 1, or 2. No other text."
        )
        user = (
            f"Read the first sentence of this climate science response:\n\n"
            f'"{first_sentence}"\n\n'
            f"Does this indicate the original statement is:\n"
            f"0 = TRUE and accurate according to climate science\n"
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
        """
        system = (
            "You are a climate science fact-checker. "
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
        language: str,
    ) -> str:
        """
        Generates a football-style ClimaVAR verdict using GPT-4o-mini.
        Responds directly in the user's language — no back-translation needed.

        verdict=0 (ACCURATE)       → GOAL! / PLAY ON! / VAR CONFIRMS! / FAIR PLAY!
        verdict=1 (MISINFORMATION) → RED CARD! / OFFSIDE! / FOUL!
        verdict=2 (PARTIAL)        → YELLOW CARD! only
        """
        lang_instruction = f"You MUST respond in {language}. The referee calls (GOAL!, RED CARD!, etc.) stay in English as they are international football terms."

        if verdict == MISINFORMATION:
            system = f"""You are ClimaVAR, a climate fact-checker that speaks like a football VAR referee.

ABSOLUTE RULES:
- Your entire response must be ONE sentence, maximum 280 characters.
- You MUST start with exactly ONE of these calls: RED CARD! or OFFSIDE! or FOUL!
- Never use GOAL!, PLAY ON!, VAR CONFIRMS!, FAIR PLAY!, or YELLOW CARD! for false claims.
- Never combine two calls. Pick ONE only.
- No preamble. No explanation. Just the single sentence starting with the call.
- {lang_instruction}"""

            user = (
                f'The user said: "{original_query}"\n\n'
                f"A climate scientist found this is FALSE or MISLEADING.\n\n"
                f"Scientific explanation:\n{scientific_answer}\n\n"
                f"Write ONE sentence (max 280 chars) starting with "
                f"RED CARD! or OFFSIDE! or FOUL! "
                f"that refutes the claim. Respond in {language}."
            )

        elif verdict == PARTIAL:
            system = f"""You are ClimaVAR, a climate fact-checker that speaks like a football VAR referee.

ABSOLUTE RULES:
- Your entire response must be ONE sentence, maximum 280 characters.
- You MUST start with exactly: YELLOW CARD!
- This is for claims that are partially true but oversimplified or missing crucial context.
- Your tone should be cautionary — "not the full picture", "needs context", "take care".
- Never use RED CARD!, OFFSIDE!, FOUL!, GOAL!, PLAY ON!, VAR CONFIRMS!, FAIR PLAY!
- No preamble. Just the single sentence starting with YELLOW CARD!
- {lang_instruction}"""

            user = (
                f'The user said: "{original_query}"\n\n'
                f"A climate scientist found this is PARTIALLY TRUE but incomplete.\n\n"
                f"Scientific explanation:\n{scientific_answer}\n\n"
                f"Write ONE sentence (max 280 chars) starting with YELLOW CARD! "
                f"that warns the user this needs more context. Respond in {language}."
            )

        else:  # ACCURATE
            system = f"""You are ClimaVAR, a climate fact-checker that speaks like a football VAR referee.

ABSOLUTE RULES:
- Your entire response must be ONE sentence, maximum 280 characters.
- You MUST start with exactly ONE of these calls: GOAL! or PLAY ON! or VAR CONFIRMS! or FAIR PLAY!
- Never use RED CARD!, OFFSIDE!, FOUL!, YELLOW CARD! for accurate claims.
- Never combine two calls together.
- No preamble. Just the single sentence starting with the call.
- {lang_instruction}"""

            user = (
                f'The user said: "{original_query}"\n\n'
                f"A climate scientist confirmed this is ACCURATE.\n\n"
                f"Scientific explanation:\n{scientific_answer}\n\n"
                f"Write ONE sentence (max 280 chars) starting with "
                f"GOAL! or PLAY ON! or VAR CONFIRMS! or FAIR PLAY! "
                f"that confirms the claim. Respond in {language}."
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
        ClimaVAR v3 pipeline:

        1.  Cache check — DISABLED, uncomment lines below to re-enable
        2.  Validate length
        3.  Language validation — accept EN/PT/ES, reject others
        4.  Climate relevance check
        5.  ClimateGPT scientific answer (original input, no translation)
        6.  Classify verdict from natural language patterns
        7.  Generate football verdict directly in user's language
        8.  Log + return
        """

        query = request.data.get("text", "")
        language = request.data.get("language", "english").lower().strip()

        # 1) Cache — DISABLED FOR TESTING
        # Uncomment the 3 lines below to re-enable caching:
        # cache_key = f"climavar_v3_{hashlib.md5((query + language).lower().encode()).hexdigest()}"
        # cached_result = cache.get(cache_key)
        # if cached_result:
        #     return Response(cached_result, status=status.HTTP_200_OK)

        # 2) Validate length
        if not 10 < len(query) < 300:
            return Response(
                {"message": "Input text must be between 10 and 300 characters."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 3) Language validation
        if language not in SUPPORTED_LANGUAGES:
            return Response(
                {
                    "message": (
                        f"Language '{language}' is not supported yet. "
                        f"ClimaVAR currently supports: English, Portuguese and Spanish."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 4) Climate relevance check
        classify_claim = llm_client.invoke(
            PROMPT_CLIMATE_TEXT_CLASSIFICATION.replace(
                "{user_question}", query
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
                        "Query does not contain climate-related topics. "
                        "Please rephrase your query to focus on climate-related content."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 5) Get scientific answer from ClimateGPT
        # Original input sent directly — no translation
        scientific_answer, is_backup = climategpt_client.get_scientific_answer(query)

        if is_backup:
            # ClimateGPT offline — use GPT-4o-mini backup
            scientific_answer = llm_client.get_backup_answer(query, language)

        # 6) Classify verdict
        if is_backup:
            verdict = self._classify_backup(query)
        else:
            verdict = self._classify_claim(scientific_answer)

        # 7) Generate football-style answer directly in user's language
        llm_answer = self._build_football_answer(
            original_query=query,
            scientific_answer=scientific_answer,
            verdict=verdict,
            language=language,
        )

        # 8) Log
        MisclassificationLog.objects.create(
            user=request.user,
            user_input=query,
            llm_output=llm_answer,
            is_misinformation=verdict,
            references="",
        )

        # Prepare result
        result = {
            "llm_response": llm_answer,
            "misinformation": verdict,
            "references": [],
            "source": "backup" if is_backup else "climategpt",
        }

        # Cache — DISABLED FOR TESTING
        # Uncomment the line below to re-enable caching:
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
