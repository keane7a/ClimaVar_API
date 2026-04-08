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
        """
        if not text:
            return True

        text_lower = text.lower().strip()
        english_words = [
            'is', 'are', 'does', 'do', 'can', 'will', 'what', 'how',
            'why', 'when', 'where', 'the', 'climate', 'global', 'warming'
        ]
        has_english_words = any(
            word in text_lower.split() for word in english_words
        )
        non_ascii = sum(1 for char in text if ord(char) > 127)
        mostly_ascii = (non_ascii / len(text)) < 0.15

        return has_english_words and mostly_ascii

    def _detect_misinformation_from_answer(self, scientific_answer: str) -> bool:
        """
        Ask GPT-4o-mini to read ClimateGPT's answer and determine if the
        original claim was misinformation.
        ClimateGPT clearly signals corrections with words like 'misconception',
        'not supported', 'inaccurate', 'hoax', 'false', 'is a claim'.
        """
        system = "You are a fact-checking assistant. Answer only YES or NO."
        user = (
            f"Read this climate science response and determine if it is "
            f"CORRECTING a false or misleading claim.\n\n"
            f"Response: {scientific_answer}\n\n"
            f"Is this response correcting misinformation? Answer only YES or NO."
        )
        result = llm_client.invoke(
            user, system=system, temperature=0.0
        ).strip().upper()
        return "YES" in result

    def _build_football_answer(
        self,
        original_query: str,
        scientific_answer: str,
        is_misinformation: bool,
    ) -> str:
        """
        Takes ClimateGPT's scientific answer and generates a football-style
        ClimaVAR verdict using GPT-4o-mini with system/user separation
        for strict instruction-following.
        """
        if is_misinformation:
            system = """You are ClimaVAR, a climate fact-checker that speaks like a football VAR referee.

ABSOLUTE RULES:
- Your entire response must be ONE sentence, maximum 280 characters.
- You MUST start with exactly ONE of these calls: RED CARD! or OFFSIDE! or FOUL! or YELLOW CARD!
- Never use GOAL!, PLAY ON!, VAR CONFIRMS!, FAIR PLAY! for false claims.
- Never combine two calls. "OFFSIDE RED CARD!" is WRONG. Pick ONE only.
- No preamble. No explanation. Just the single sentence starting with the call."""

            user = (
                f'The user said: "{original_query}"\n\n'
                f"A climate scientist reviewed this and found it is FALSE or MISLEADING. "
                f"Here is the scientific explanation:\n\n{scientific_answer}\n\n"
                f"Now write ONE sentence (max 280 chars) starting with "
                f"RED CARD! or OFFSIDE! or FOUL! or YELLOW CARD! "
                f"that refutes the claim using the scientific evidence above."
            )
        else:
            system = """You are ClimaVAR, a climate fact-checker that speaks like a football VAR referee.

ABSOLUTE RULES:
- Your entire response must be ONE sentence, maximum 280 characters.
- You MUST start with exactly ONE of these calls: GOAL! or PLAY ON! or VAR CONFIRMS! or FAIR PLAY!
- Never use RED CARD!, OFFSIDE!, FOUL!, YELLOW CARD! for true statements or questions.
- Never combine two calls together.
- No preamble. No explanation. Just the single sentence starting with the call."""

            user = (
                f'The user said: "{original_query}"\n\n'
                f"A climate scientist reviewed this and confirmed it is ACCURATE "
                f"or answered the question. Here is the scientific explanation:\n\n"
                f"{scientific_answer}\n\n"
                f"Now write ONE sentence (max 280 chars) starting with "
                f"GOAL! or PLAY ON! or VAR CONFIRMS! or FAIR PLAY! "
                f"that confirms or answers using the scientific evidence above."
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

        1. Cache check — instant for repeat queries
        2. Validate input length
        3. Smart English detection — skip translation if not needed
        4. Climate relevance check
        5. ClimateGPT — scientific answer + references
        6. Detect misinformation from ClimateGPT answer
        7. GPT-4o-mini — football-style verdict
        8. Translate back if needed
        9. Log to database (is_misinformation column preserved)
        10. Cache + return
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

        # 3) Smart translation — skip if English
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
        if "0" in classify_claim[-10:]:
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

        # 5) Get scientific answer from ClimateGPT
        scientific_answer, references = climategpt_client.get_scientific_answer(
            query_english
        )

        # 6) Detect misinformation from ClimateGPT's answer
        is_misinformation = self._detect_misinformation_from_answer(
            scientific_answer
        )

        # 7) Generate football-style answer
        llm_answer = self._build_football_answer(
            original_query=query_english,
            scientific_answer=scientific_answer,
            is_misinformation=is_misinformation,
        )

        # 8) Translate back if needed
        if src_lang != "English":
            final_answer = llm_client.translate_language(llm_answer, src_lang)[1]
        else:
            final_answer = llm_answer

        # 9) Log — is_misinformation column preserved exactly as before
        MisclassificationLog.objects.create(
            user=request.user,
            user_input=request.data.get("text", ""),
            llm_output=final_answer,
            is_misinformation=is_misinformation,
            references="\n".join(references),
        )

        # 10) Cache + return
        result = {
            "llm_response": final_answer,
            "misinformation": int(is_misinformation),
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
