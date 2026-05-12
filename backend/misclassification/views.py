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

# Referee calls per language
REFEREE_CALLS = {
    "english": {
        "accurate": "GOAL!",
        "misinformation": "RED CARD!",
        "partial": "YELLOW CARD!",
    },
    "portuguese": {
        "accurate": "GOL!",
        "misinformation": "CARTÃO VERMELHO!",
        "partial": "CARTÃO AMARELO!",
    },
    "spanish": {
        "accurate": "GOL!",
        "misinformation": "TARJETA ROJA!",
        "partial": "TARJETA AMARILLA!",
    },
}

# Init GPT-4o-mini
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm_client = LLMClient(openai_client, model="gpt-4o-mini", temperature=0.2)

# Init ClimateGPT
climategpt_client = ClimateGPTClient(api_key=os.getenv("CLIMATEGPT_API_KEY"))


class MisclassificationViewSet(viewsets.ViewSet):
    permission_classes = [IsAdminUser]

    def _classify_claim(self, query: str, scientific_answer: str) -> int:
        """
        Classify verdict by comparing the original query against the full
        ClimateGPT response using GPT-4o-mini.

        When ClimateGPT agrees with a claim (verdict=0), performs a second
        sanity check to catch cases where ClimateGPT itself oversimplified
        or missed important nuance (e.g. nuclear lifecycle emissions).

        Returns:
            0 = ACCURATE
            1 = MISINFORMATION
            2 = PARTIAL / needs context
        """
        system = (
            "You are a climate fact-checking assistant. "
            "Reply with exactly ONE digit: 0, 1, or 2. No other text."
        )
        user = (
            f"A user submitted the following climate claim or question:\n"
            f"\"{query}\"\n\n"
            f"A climate scientist responded with:\n"
            f"\"{scientific_answer}\"\n\n"
            f"Based on the FULL scientist response, classify the original claim:\n\n"
            f"0 = ACCURATE: The scientist confirms the claim and it is fully "
            f"supported by climate science.\n\n"
            f"1 = MISINFORMATION: The claim is fundamentally wrong and directly "
            f"contradicts scientific consensus. Reserve for claims that are "
            f"straightforwardly false with no significant truth in them.\n\n"
            f"2 = PARTIAL: The claim contains some truth but the conclusion or "
            f"framing is misleading, oversimplified, or missing crucial context. "
            f"Use when a true fact leads to a false or exaggerated conclusion.\n\n"
            f"CRITICAL RULE: If the claim mixes a true fact with a false or "
            f"misleading conclusion, classify as 2 not 1.\n\n"
            f"EXAMPLES:\n"
            f"Claim: 'Climate change is not caused by humans' → 1\n"
            f"Claim: 'Global warming is a hoax' → 1\n"
            f"Claim: 'CO2 is only 0.04% so cannot cause warming' → 2\n"
            f"Claim: 'Developing countries are the main cause of emissions' → 2\n"
            f"Claim: 'Nuclear energy produces no emissions' → 2\n"
            f"Claim: 'Electric vehicles produce zero emissions' → 2\n"
            f"Claim: 'Renewable energy is already cheaper everywhere' → 2\n"
            f"Claim: 'Greenhouse gases reached record levels in 2023' → 0\n"
            f"Claim: 'Human activities are the primary cause of climate change' → 0\n\n"
            f"Reply with only 0, 1, or 2."
        )
        result = llm_client.invoke(
            user, system=system, temperature=0.0
        ).strip()

        initial_verdict = ACCURATE
        for char in result:
            if char in ("0", "1", "2"):
                initial_verdict = int(char)
                break

        # Second sanity check — only when initial verdict is ACCURATE (0)
        # Adds ~1 second only for GOAL verdicts
        # Catches cases where ClimateGPT agreed with an oversimplified claim
        if initial_verdict == ACCURATE:
            sanity_system = (
                "You are a climate science expert. "
                "Reply with exactly ONE digit: 0 or 2. No other text."
            )
            sanity_user = (
                f"Is this climate claim fully accurate, or does it oversimplify, "
                f"exaggerate, or omit crucial context that would change its meaning?\n\n"
                f"Claim: \"{query}\"\n\n"
                f"0 = Fully accurate — no important caveats or missing context\n"
                f"2 = Oversimplified or missing crucial context — partially true "
                f"but would give someone a misleading impression\n\n"
                f"EXAMPLES:\n"
                f"'Human activities are the primary cause of climate change' → 0\n"
                f"'Greenhouse gases reached record levels in 2023' → 0\n"
                f"'Sea levels are rising due to climate change' → 0\n"
                f"'Arctic ice is melting faster than predicted' → 0\n"
                f"'Nuclear energy produces no emissions' → 2 "
                f"(has very low but non-zero lifecycle emissions)\n"
                f"'Electric vehicles produce zero emissions' → 2 "
                f"(zero tailpipe but not zero lifecycle)\n"
                f"'Renewable energy is cheaper than fossil fuels everywhere' → 2 "
                f"(true in many but not all contexts)\n"
                f"'Nuclear energy should be the primary solution to climate change "
                f"because it produces no emissions' → 2 "
                f"(overstates nuclear role and misrepresents emissions)\n\n"
                f"Reply with only 0 or 2."
            )
            sanity_result = llm_client.invoke(
                sanity_user, system=sanity_system, temperature=0.0
            ).strip()

            for char in sanity_result:
                if char in ("0", "2"):
                    return int(char)

        return initial_verdict

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
            f"\"{query}\"\n\n"
            f"0 = ACCURATE: Correct and fully supported by scientific consensus.\n"
            f"1 = MISINFORMATION: Fundamentally false, no significant truth in it.\n"
            f"2 = PARTIAL: Contains some truth but misleading conclusion or "
            f"missing crucial context. Use when a true fact leads to a false conclusion.\n\n"
            f"EXAMPLES:\n"
            f"'Greenhouse gas concentrations reached record levels in 2023' → 0\n"
            f"'Human activities are the primary cause of climate change' → 0\n"
            f"'Climate change is not caused by human activity' → 1\n"
            f"'Global warming is a hoax invented by scientists' → 1\n"
            f"'Electric vehicles produce zero emissions' → 2\n"
            f"'Nuclear energy produces no emissions' → 2\n"
            f"'Developing countries are the main cause of rising emissions' → 2\n\n"
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
        Responds directly in the user's language.
        Referee calls are translated based on language:

        ACCURATE      → GOAL! / GOL! / GOL!
        MISINFORMATION → RED CARD! / CARTÃO VERMELHO! / TARJETA ROJA!
        PARTIAL       → YELLOW CARD! / CARTÃO AMARELO! / TARJETA AMARILLA!
        """
        calls = REFEREE_CALLS.get(language, REFEREE_CALLS["english"])

        if verdict == MISINFORMATION:
            call = calls["misinformation"]
            system = (
                f"You are ClimaVAR, a climate fact-checker that speaks like a football VAR referee. "
                f"You MUST respond entirely in {language}. "
                f"Your entire response must be ONE sentence, maximum 280 characters. "
                f"You MUST start with exactly: {call} "
                f"Never use any other call. No preamble. "
                f"Just the single sentence starting with {call}"
            )
            user = (
                f'The user said: "{original_query}"\n\n'
                f"A climate scientist found this is FALSE or MISLEADING.\n\n"
                f"Scientific explanation:\n{scientific_answer}\n\n"
                f"Write ONE sentence (max 280 chars) starting with {call} "
                f"that refutes the claim. You MUST respond in {language}."
            )

        elif verdict == PARTIAL:
            call = calls["partial"]
            system = (
                f"You are ClimaVAR, a climate fact-checker that speaks like a football VAR referee. "
                f"You MUST respond entirely in {language}. "
                f"Your entire response must be ONE sentence, maximum 280 characters. "
                f"You MUST start with exactly: {call} "
                f"This is for claims that are partially true but misleading or missing crucial context. "
                f"Your tone should be cautionary. "
                f"Never use any other call. No preamble. "
                f"Just the single sentence starting with {call}"
            )
            user = (
                f'The user said: "{original_query}"\n\n'
                f"A climate scientist found this is PARTIALLY TRUE but incomplete or misleading.\n\n"
                f"Scientific explanation:\n{scientific_answer}\n\n"
                f"Write ONE sentence (max 280 chars) starting with {call} "
                f"warning the user this needs more context. You MUST respond in {language}."
            )

        else:  # ACCURATE
            call = calls["accurate"]
            system = (
                f"You are ClimaVAR, a climate fact-checker that speaks like a football VAR referee. "
                f"You MUST respond entirely in {language}. "
                f"Your entire response must be ONE sentence, maximum 280 characters. "
                f"You MUST start with exactly: {call} "
                f"Never use any other call. No preamble. "
                f"Just the single sentence starting with {call}"
            )
            user = (
                f'The user said: "{original_query}"\n\n'
                f"A climate scientist confirmed this is ACCURATE.\n\n"
                f"Scientific explanation:\n{scientific_answer}\n\n"
                f"Write ONE sentence (max 280 chars) starting with {call} "
                f"that confirms the claim. You MUST respond in {language}."
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

        1.  Cache — DISABLED, uncomment lines to re-enable
        2.  Validate length
        3.  Language validation — EN/PT/ES only
        4.  Climate relevance check
        5.  ClimateGPT scientific answer (original input, no translation)
        6.  Classify verdict using full query vs full response
            + sanity check for ACCURATE verdicts
        7.  Generate football verdict in user's language with translated calls
        8.  Log + return
        """

        query = request.data.get("text", "")
        language = request.data.get("language", "english").lower().strip()

        # 1) Cache — DISABLED FOR TESTING
        # To re-enable, uncomment these 3 lines:
        # cache_key = f"climavar_v3_{hashlib.md5((query+language).lower().encode()).hexdigest()}"
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
            scientific_answer = llm_client.get_backup_answer(query, language)

        # 6) Classify verdict
        # Includes sanity check for ACCURATE verdicts
        if is_backup:
            verdict = self._classify_backup(query)
        else:
            verdict = self._classify_claim(query, scientific_answer)

        # 7) Generate football answer in user's language with translated calls
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

        result = {
            "llm_response": llm_answer,
            "misinformation": verdict,
            "references": [],
            "source": "backup" if is_backup else "climategpt",
        }

        # Cache — DISABLED FOR TESTING
        # To re-enable, uncomment this line:
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
