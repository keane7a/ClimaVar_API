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

        Reads the FULL response — not just the first sentence — so it can
        correctly interpret cases where ClimateGPT implicitly contradicts
        a claim using affirmative language rather than explicit FALSE signals.

        Works across English, Portuguese and Spanish.

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
            f"0 = ACCURATE: The scientist's response confirms or agrees with "
            f"the claim. The claim is supported by climate science.\n\n"
            f"1 = MISINFORMATION: The scientist's response contradicts, refutes, "
            f"or corrects the claim — even if it does not use the word 'false' "
            f"explicitly. If the scientist explains what is actually true and it "
            f"differs from the claim, classify as 1.\n\n"
            f"2 = PARTIAL: The scientist's response says the claim is partially "
            f"true but oversimplified, exaggerated, or missing important context "
            f"that changes its meaning.\n\n"
            f"IMPORTANT: Focus on whether the scientist's response AGREES or "
            f"DISAGREES with the original claim, not on the language used. "
            f"A response that explains the opposite of what the claim states "
            f"is a contradiction and should be classified as 1.\n\n"
            f"EXAMPLES:\n"
            f"Claim: 'Climate change is not caused by humans'\n"
            f"Response: 'Climate change is intrinsically linked to human activities...'\n"
            f"→ 1 (scientist contradicts the claim)\n\n"
            f"Claim: 'Greenhouse gases reached record levels in 2023'\n"
            f"Response: '2023 marked a significant milestone with record greenhouse gas levels...'\n"
            f"→ 0 (scientist confirms the claim)\n\n"
            f"Claim: 'Electric vehicles produce zero emissions'\n"
            f"Response: 'While EVs produce no tailpipe emissions, manufacturing and grid emissions must be considered...'\n"
            f"→ 2 (partially true but oversimplified)\n\n"
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
        Uses GPT-4o-mini's own knowledge of scientific consensus.
        """
        system = (
            "You are a climate science fact-checker. "
            "Reply with exactly ONE digit: 0, 1, or 2. No other text."
        )
        user = (
            f"Classify this climate statement based on scientific consensus:\n\n"
            f"\"{query}\"\n\n"
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
        lang_instruction = (
            f"You MUST respond in {language}. "
            f"The referee calls (GOAL!, RED CARD!, YELLOW CARD!, etc.) "
            f"always stay in English as they are international football terms."
        )

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

        1.  Cache check — DISABLED, uncomment lines to re-enable
        2.  Validate length
        3.  Language validation — accept EN/PT/ES, reject others
        4.  Climate relevance check
        5.  ClimateGPT scientific answer (original input, no translation)
        6.  Classify verdict by comparing full query vs full response
        7.  Generate football verdict directly in user's language
        8.  Log + return
        """

        query = request.data.get("text", "")
        language = request.data.get("language", "english").lower().strip()

        # 1) Cache — DISABLED FOR TESTING
        # To re-enable caching, uncomment these 3 lines:
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
        # Compare full query vs full ClimateGPT response
        if is_backup:
            verdict = self._classify_backup(query)
        else:
            verdict = self._classify_claim(query, scientific_answer)

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
        # To re-enable caching, uncomment this line:
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
