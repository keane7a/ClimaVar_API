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
from misclassification.utils.rag import (
    LLMClient,
    ClimateGPTClient,
    SUPPORTED_LANGUAGES,
    is_arabic,
    language_label,
)
from misclassification.utils.prompts import *
import os
from django.core.cache import cache
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    "arabic": {
        "accurate": "هدف!",
        "misinformation": "بطاقة حمراء!",
        "partial": "بطاقة صفراء!",
    },
    "turkish": {
        "accurate": "GOL!",
        "misinformation": "KIRMIZI KART!",
        "partial": "SARI KART!",
    },
}

# Init GPT-4o-mini
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm_client = LLMClient(openai_client, model="gpt-4o-mini", temperature=0.2)

# Init ClimateGPT
climategpt_client = ClimateGPTClient(api_key=os.getenv("CLIMATEGPT_API_KEY"))


class MisclassificationViewSet(viewsets.ViewSet):
    permission_classes = [IsAdminUser]

    def _detect_language(self, query: str):
        """
        Detect the language the user typed the claim in.
        The answer must follow the INPUT language, not the interface language.
        Returns one of SUPPORTED_LANGUAGES, or None if unknown/unsupported.
        """
        if is_arabic(query):
            return "arabic"
        system = (
            "Identify the language of the user's text. "
            "Reply with exactly one word: english, portuguese, spanish, "
            "arabic, turkish, or other. No other text."
        )
        try:
            result = llm_client.invoke(query, system=system, temperature=0.0)
            result = result.strip().lower()
        except Exception as e:
            print(f"[LANG DETECT ERROR] {type(e).__name__}: {e}", flush=True)
            return None
        for lang in SUPPORTED_LANGUAGES:
            if lang in result:
                return lang
        return None

    def _classify_claim(self, query: str, scientific_answer: str) -> int:
        """
        Classify verdict by comparing the original query against the full
        ClimateGPT response using GPT-4o-mini.

        When ClimateGPT agrees with a claim (verdict=0), performs a second
        sanity check, UNLESS the input is a question, in which case
        the sanity check is skipped (questions are not oversimplified claims).
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

        # TEMPORARY DEBUG LOG: what ClimateGPT said and the first verdict
        print(
            f"[VERDICT DEBUG] climategpt_answer={scientific_answer[:400]!r}",
            flush=True,
        )
        print(f"[VERDICT DEBUG] initial_verdict={initial_verdict}", flush=True)

        # Second sanity check: only for ACCURATE verdicts on non-questions
        # Questions are skipped since they are not claims that can be oversimplified
        if initial_verdict == ACCURATE and not llm_client.is_question(query):
            sanity_system = (
                "You are a climate science expert. "
                "Reply with exactly ONE digit: 0 or 2. No other text."
            )
            sanity_user = (
                f"Check this climate claim for oversimplification.\n\n"
                f"Claim: \"{query}\"\n\n"
                f"Answer 2 ONLY if the claim itself says something misleading: "
                f"an absolute word that makes it false (zero, no, none, all, "
                f"never, always, only, alone, everywhere), a clear exaggeration, "
                f"or a false conclusion drawn from a true fact.\n\n"
                f"Answer 0 if the claim matches established scientific findings "
                f"(for example IPCC conclusions), even if more background could "
                f"be added. Missing extra detail, nuance or background is NOT a "
                f"reason to answer 2.\n\n"
                f"0 = Accurate as stated\n"
                f"2 = Misleading as stated\n\n"
                f"EXAMPLES:\n"
                f"'Human activities are the primary cause of climate change' → 0\n"
                f"'Greenhouse gases reached record levels in 2023' → 0\n"
                f"'Sea levels are rising due to climate change' → 0\n"
                f"'Global average temperature has risen about 1.1°C since "
                f"pre-industrial times' → 0\n"
                f"'Ocean acidity has increased since the Industrial Revolution' → 0\n"
                f"'The 2010s were the warmest decade on record' → 0\n"
                f"'Nuclear energy produces no emissions' → 2\n"
                f"'Electric vehicles produce zero emissions' → 2\n"
                f"'Renewable energy is cheaper than fossil fuels everywhere' → 2\n"
                f"'Nuclear energy should be the primary solution to climate change "
                f"because it produces no emissions' → 2\n\n"
                f"Reply with only 0 or 2."
            )
            sanity_result = llm_client.invoke(
                sanity_user, system=sanity_system, temperature=0.0
            ).strip()

            # TEMPORARY DEBUG LOG: sanity check result
            print(f"[VERDICT DEBUG] sanity_result={sanity_result!r}", flush=True)

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
            f"missing crucial context.\n\n"
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
        Referee calls translated per language.
        """
        calls = REFEREE_CALLS.get(language, REFEREE_CALLS["english"])

        # Name used in prompts (Arabic becomes "Modern Standard Arabic")
        lang = language_label(language)

        if verdict == MISINFORMATION:
            call = calls["misinformation"]
            system = (
                f"You are ClimaVAR, a climate fact-checker that speaks like a football VAR referee. "
                f"You MUST respond entirely in {lang}. "
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
                f"that refutes the claim. You MUST respond in {lang}."
            )

        elif verdict == PARTIAL:
            call = calls["partial"]
            system = (
                f"You are ClimaVAR, a climate fact-checker that speaks like a football VAR referee. "
                f"You MUST respond entirely in {lang}. "
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
                f"warning the user this needs more context. You MUST respond in {lang}."
            )

        else:  # ACCURATE
            call = calls["accurate"]
            system = (
                f"You are ClimaVAR, a climate fact-checker that speaks like a football VAR referee. "
                f"You MUST respond entirely in {lang}. "
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
                f"that confirms the claim. You MUST respond in {lang}."
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
        ClimaVAR v3 pipeline (optimized, with Arabic support):

        1.  Cache check (24h TTL)
        2.  Validate length
        3.  Language: detected from the INPUT text (EN/PT/ES/AR/TR), run in
            parallel with step 4. The interface language sent by the
            client is only a fallback when detection fails.
        3b. Original text (Arabic included) is sent to ClimateGPT
            and classifiers, with no translation
        4.  Climate relevance check + ClimateGPT, run IN PARALLEL
        5.  Classify verdict (full response comparison + sanity check
            for non-question ACCURATE verdicts)
        6.  Generate football verdict in user's language
        6b. Final language check (Arabic)
        7.  Log + cache + return
        """

        query = request.data.get("text", "")
        # Language sent by the client (interface language). Used only as a
        # fallback when the input language cannot be detected.
        requested_language = (
            request.data.get("language", "english") or "english"
        ).lower().strip()

        # 1) Cache check
        cache_key = f"climavar_v3_{hashlib.md5((query + requested_language).lower().encode()).hexdigest()}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return Response(cached_result, status=status.HTTP_200_OK)

        # 2) Validate length
        if not 10 < len(query) < 300:
            return Response(
                {"message": "Input text must be between 10 and 300 characters."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 3b) All languages (Arabic included) are sent to ClimateGPT and the
        # classifiers in the original text. The new ClimateGPT model
        # (climategpt4-moe36-int4) handles Arabic directly, so the previous
        # Arabic-to-English translation step was removed.
        query_for_model = query

        # 4) Run language detection, climate relevance check AND ClimateGPT
        # IN PARALLEL
        with ThreadPoolExecutor(max_workers=3) as executor:
            language_future = executor.submit(self._detect_language, query)
            relevance_future = executor.submit(
                llm_client.is_climate_related, query_for_model
            )
            climategpt_future = executor.submit(
                climategpt_client.get_scientific_answer, query_for_model
            )

            detected_language = language_future.result()
            is_climate_related = relevance_future.result()
            scientific_answer, is_backup = climategpt_future.result()

        # 3) Language decision: the answer follows the INPUT language.
        # The interface language is only a fallback.
        language = detected_language or requested_language

        # TEMPORARY DEBUG LOG: language decision
        print(
            f"[LANG DEBUG] detected={detected_language} "
            f"requested={requested_language} used={language}",
            flush=True,
        )

        if language not in SUPPORTED_LANGUAGES:
            return Response(
                {
                    "message": (
                        f"Language '{language}' is not supported yet. "
                        f"ClimaVAR currently supports: English, Portuguese, Spanish, Arabic and Turkish."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not is_climate_related:
            return Response(
                {
                    "message": (
                        "Query does not contain climate-related topics. "
                        "Please rephrase your query to focus on climate-related content."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if is_backup:
            # Backup (GPT-4o-mini) receives the original text and handles Arabic
            scientific_answer = llm_client.get_backup_answer(query, language)

        # 5) Classify verdict
        if is_backup:
            verdict = self._classify_backup(query_for_model)
        else:
            verdict = self._classify_claim(query_for_model, scientific_answer)

        # 6) Generate football answer in user's language
        llm_answer = self._build_football_answer(
            original_query=query,
            scientific_answer=scientific_answer,
            verdict=verdict,
            language=language,
        )

        # 6b) Final language check: translates to Arabic only if needed
        llm_answer = llm_client.ensure_language(llm_answer, language)

        # 7) Log + cache + return
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
