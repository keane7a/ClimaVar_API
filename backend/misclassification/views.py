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
        """
        Detects whether text is English.
        Strategy:
        1. If text contains accented characters common in PT/ES, it is NOT English.
        2. If text contains uniquely English function words, it IS English.
        3. Otherwise assume non-English and translate to be safe.
        """
        if not text:
            return True

        # Step 1 — accented chars that strongly indicate PT/ES/FR
        non_english_chars = set('àáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿãõç')
        if any(char in non_english_chars for char in text.lower()):
            return False

        # Step 2 — uniquely English function words
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

        # Step 3 — uncertain, translate to be safe
        return False

    def _is_negation(self, text: str) -> bool:
        """
        Detects whether the statement is a negation or denial.
        Used to build a more direct ClimateGPT query for clear denials.
        """
        text_lower = text.lower()
        negation_patterns = [
            'is not', 'are not', 'does not', 'do not', 'cannot', 'can not',
            'isn\'t', 'aren\'t', 'doesn\'t', 'don\'t', 'wasn\'t', 'weren\'t',
            'not caused', 'not real', 'not happening', 'not true', 'not exist',
            'no evidence', 'never', 'hoax', 'fake', 'fraud', 'myth',
            'nothing to do', 'has nothing', 'have nothing',
        ]
        return any(pattern in text_lower for pattern in negation_patterns)

    def _classify_claim(self, scientific_answer: str) -> int:
        """
        Ask GPT-4o-mini to classify the claim into one of three categories
        based on ClimateGPT's scientific answer.

        0 = ACCURATE — claim is correct per scientific consensus
        1 = MISINFORMATION — claim is false or clearly misleading
        2 = PARTIAL — claim is misleading or oversimplified in a way
                      that fundamentally changes its meaning

        Key principle: scientific answers often add nuance even to
        accurate statements. A broadly correct claim should be 0,
        not 2, even if the answer mentions additional factors.
        Only use 2 when the original claim itself would mislead someone.
        """
        system = (
            "You are a climate fact-checking assistant. "
            "You must reply with exactly ONE digit: 0, 1, or 2. No other text."
        )
        user = (
            f"Read this climate science response carefully:\n\n"
            f"{scientific_answer}\n\n"
            f"Classify the original statement using these rules:\n\n"
            f"0 = ACCURATE: The statement is correct and aligns with scientific "
            f"consensus. Use 0 even if the scientific answer adds nuance or mentions "
            f"other contributing factors — nuance in the answer does NOT make the "
            f"original claim inaccurate. When in doubt between 0 and 2, choose 0.\n\n"
            f"1 = MISINFORMATION: The statement is clearly FALSE or directly "
            f"contradicts scientific consensus. Use 1 when the scientific response "
            f"explicitly corrects or refutes the claim. Also use 1 for statements "
            f"that deny, negate, or dismiss established climate science facts.\n\n"
            f"2 = PARTIAL: Use 2 ONLY when the original statement itself is "
            f"misleading or oversimplified in a way that would cause "
            f"misunderstanding — for example, a statement that is technically "
            f"partially true but omits a crucial fact that reverses its meaning, "
            f"or a statement that mixes true and false elements together.\n\n"
            f"CALIBRATION EXAMPLES:\n"
            f"- 'CO2 is the main driver of global warming' → 0 (accurate, even "
            f"if other gases also contribute)\n"
            f"- 'Greenhouse gas concentrations reached record levels in 2023' → 0 "
            f"(confirmed by WMO, straightforwardly accurate)\n"
            f"- 'Global warming is caused mainly by human CO2 emissions' → 0 "
            f"(core scientific consensus, accurate)\n"
            f"- 'Human CO2 emissions cause global warming' → 0 (accurate)\n"
            f"- 'Climate change is not caused by humans' → 1 (clear misinformation, "
            f"directly contradicts consensus)\n"
            f"- 'Climate change is not caused by human activity' → 1 "
            f"(clear misinformation, denial of established science)\n"
            f"- 'Global warming is not happening' → 1 (clear misinformation)\n"
            f"- 'Climate change is a hoax invented by scientists' → 1 "
            f"(clear misinformation)\n"
            f"- 'Electric vehicles produce zero emissions' → 2 (partial — true for "
            f"tailpipe but ignores manufacturing and grid emissions, misleading)\n"
            f"- 'Humans emit a tiny fraction of CO2 compared to nature, so we are "
            f"not responsible for warming' → 2 (mixes a partial truth with a false "
            f"conclusion in a misleading way)\n"
            f"- 'Planting trees alone can solve climate change' → 2 (oversimplified "
            f"in a way that causes misunderstanding)\n\n"
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
        3.  Smart English detection — accent-based + function word check
        4.  Climate relevance check — robust Output: 0/1 parsing
        5.  Detect question vs statement
        6.  Build context-aware ClimateGPT query
            — negations get a more direct "correct or incorrect?" framing
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
        # cache_key = f"climavar_query_{hashlib.md5(query.lower().encode()).hexdigest()}"
        # cached_result = cache.get(cache_key)
        # if cached_result:
        #    return Response(cached_result, status=status.HTTP_200_OK)

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
        # Questions → sent as-is for direct answer
        # Negations → ask explicitly if claim is correct or incorrect
        # Statements → framed as accuracy check
        if not is_statement:
            climategpt_query = query_english
        elif self._is_negation(query_english):
            climategpt_query = (
                f"Is the following claim correct or incorrect according to "
                f"climate science? Please state clearly whether it is true or "
                f"false: {query_english}"
            )
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
