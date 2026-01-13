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
from misclassification.utils.rag import LLMClient, CARDSClient
from misclassification.utils.embeddings import EmbeddingModel
import chromadb
import re
import os
from concurrent.futures import ThreadPoolExecutor


# Init LLM Model
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm_client = LLMClient(openai_client, model="gpt-4o-mini", temperature=0.2)

# Init CARDS Model
cards_client = OpenAI(
    api_key=os.getenv("CARDS_API_KEY"), base_url=os.getenv("CARDS_BASE_URL")
)
cards_client = CARDSClient(
    cards_client, model="cards-mini-sonnet-2024-12-05", temperature=0.0
)

# Init embedding model and ChromaDB
chromadb_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database="ClimaVAR",
)

embedding_model = EmbeddingModel(
    embedding_model="text-embedding-3-small",
    openai_client=openai_client,
    chromadb_client=chromadb_client,
    collection_name="ClimaVAR_v2",
)


class MisclassificationViewSet(viewsets.ViewSet):
    permission_classes = [IsAdminUser]

    def _get_evidence_block(self, query, top_k=2):
        """
        Get evidence block from chromadb based on query.
        Optimized: top_k=2, max_char=350 for faster performance.
        """
        max_char = 350
        results = embedding_model.query_chromadb(query, top_k)
        if not results:
            return "- (no evidence available)", []

        lines, cites = [], []
        for r in results:
            text = r.get("text", "").strip()
            text = re.sub(r"\s+", " ", text)
            text = text[:max_char] + "...\n"
            lines.append(f"- {text}")

            title = r.get("title", "")
            year = r.get("year", "")
            url = r.get("url", "")
            cites.append(f"{title}, ({year}) - {url}")

        return "".join(lines), list(set(cites))

    def _is_likely_english(self, text):
        """
        Quick heuristic to detect if text is likely English.
        Skips expensive translation call if true.
        """
        if not text:
            return True
        non_ascii = sum(1 for char in text if ord(char) > 127)
        return non_ascii / len(text) < 0.15

    def _check_climate_relevance(self, query):
        """
        Check if query is climate-related using LLM.
        More lenient to avoid false negatives.
        """
        prompt = f"""Determine if this text is related to climate science, environmental science, or climate change.

Text: "{query}"

Climate-related topics include:
- Climate change, global warming, greenhouse gases, carbon dioxide
- Temperature, weather patterns, sea levels, ice, oceans, glaciers
- Renewable energy, fossil fuels, coal, oil (in climate/environment context)
- Deforestation, biodiversity, ecosystems, species extinction
- Climate policy, IPCC, COP conferences, Paris Agreement
- Environmental impacts of human activities
- Climate models, climate data, climate science

NOT climate-related:
- Pure financial topics without climate connection
- General news without climate angle
- Spam, gibberish, or nonsense text

Think carefully: Does this relate to climate or environment in ANY way?

Answer with ONLY "1" if climate-related, or "0" if not.

Output:"""

        response = llm_client.invoke(prompt, temperature=0.0).strip()
        return "1" in response

    def _build_prompt_for_question(self, user_question, evidence_block):
        """
        Build prompt for answering QUESTIONS.
        Uses positive referee calls.
        """
        prompt = f"""You are ClimaVAR, a climate fact-checker using football referee language.

QUESTION: "{user_question}"

SCIENTIFIC EVIDENCE:
{evidence_block}

YOUR TASK:
Answer this question accurately in ONE sentence (max 280 characters).

REFEREE CALLS (pick ONE that fits):
- "GOAL!" - for confirming clear facts
- "PLAY ON!" - for straightforward answers
- "VAR CONFIRMS!" - for fact-checked information
- "FAIR PLAY!" - for balanced, accurate statements

IMPORTANT RULES:
1. Use ONLY ONE referee call at the start
2. Keep answer under 280 characters
3. Base answer on the evidence provided
4. Be friendly and clear
5. No preamble, just: [CALL] [Answer]

EXAMPLE:
Question: "Are sea levels rising?"
Response: "GOAL! Sea levels are rising at 4mm per year, more than double the 20th-century rate, driven by warming oceans and melting ice!"

Now answer the user's question:"""

        return prompt

    def _build_prompt_for_false_claim(self, user_claim, evidence_block, categories):
        """
        Build prompt for refuting FALSE claims.
        Uses negative referee calls - ONE ONLY.
        """
        prompt = f"""You are ClimaVAR, a climate fact-checker using football referee language.

CLAIM (identified as MISINFORMATION): "{user_claim}"
Misinformation categories: {categories}

SCIENTIFIC EVIDENCE:
{evidence_block}

YOUR TASK:
Refute this false claim in ONE sentence (max 280 characters).

REFEREE CALLS (pick ONLY ONE):
- "RED CARD!" - for serious misinformation
- "OFFSIDE!" - for incorrect claims  
- "FOUL!" - for misleading statements

CRITICAL RULES:
1. Use ONLY ONE referee call (NOT "OFFSIDE RED CARD!" or combinations)
2. Keep under 280 characters
3. Use evidence to explain why it's wrong
4. Be direct but friendly
5. No preamble, just: [ONE CALL] [Refutation]

EXAMPLE:
Claim: "Climate change is a hoax"
Response: "RED CARD! That's misinformation—99% of climate scientists agree climate change is real and human-caused, backed by decades of data!"

Now refute the user's claim:"""

        return prompt

    def _build_prompt_for_true_statement(self, user_statement, evidence_block):
        """
        Build prompt for confirming TRUE/ACCURATE statements.
        Uses positive referee calls.
        """
        prompt = f"""You are ClimaVAR, a climate fact-checker using football referee language.

STATEMENT (verified as accurate): "{user_statement}"

SCIENTIFIC EVIDENCE:
{evidence_block}

YOUR TASK:
Confirm this accurate statement in ONE sentence (max 280 characters).

REFEREE CALLS (pick ONE):
- "GOAL!" - for correct statements
- "PLAY ON!" - for accurate claims
- "VAR CONFIRMS!" - for verified facts
- "FAIR PLAY!" - for honest, accurate assessments

IMPORTANT RULES:
1. Use ONLY ONE referee call at the start
2. Keep under 280 characters
3. Add helpful context from evidence
4. Be supportive and friendly
5. No preamble, just: [CALL] [Confirmation]

EXAMPLE:
Statement: "Sea levels are rising"
Response: "PLAY ON! Spot on—sea levels are rising at 4mm annually, double the 20th-century rate, due to warming oceans and melting ice!"

Now confirm the user's statement:"""

        return prompt

    @extend_schema(
        summary="Check Misclassification",
        description="Evaluate a user-supplied climate statement and return an LLM response in a football referee manner.",
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
        ClimaVAR V2 Pipeline - Optimized for quality and performance.
        
        Flow:
        1. Validate input length
        2. Smart translation (skip if English)
        3. Check climate relevance
        4. Branch based on input type:
           - QUESTIONS: Get evidence → Answer directly (no CARDS)
           - STATEMENTS: Get evidence + CARDS in parallel → Respond based on result
        5. Translate response back if needed
        6. Log and return
        """

        query = request.data.get("text", "")

        # 1) Validate length
        if not 10 < len(query) < 300:
            return Response(
                {"message": "Input text must be between 10 and 300 characters."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 2) Smart translation - skip if English (saves 1-2 seconds)
        if self._is_likely_english(query):
            src_lang = "English"
            query_english = query
        else:
            src_lang, query_english = llm_client.translate_language(query, "English")
            if not (query_english and src_lang):
                return Response(
                    {"message": "Unsupported language for translation."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        # 3) Check climate relevance (improved to reduce false negatives)
        is_climate_related = self._check_climate_relevance(query_english)
        
        if not is_climate_related:
            return Response(
                {
                    "message": "Query does not contain climate-related topics or sufficient climate-related keywords. Please rephrase your query to focus on climate-related content."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 4) Determine if question or statement
        is_question = "?" in query_english

        if is_question:
            # ===== QUESTION PATH =====
            # Questions don't need CARDS - just answer from evidence
            
            evidence_block, cites = self._get_evidence_block(query_english, top_k=2)
            
            prompt = self._build_prompt_for_question(query_english, evidence_block)
            
            llm_answer = llm_client.invoke(prompt, temperature=0.0)
            
            is_misinformation = False
            categories = ""

        else:
            # ===== STATEMENT PATH =====
            # Run CARDS + Evidence IN PARALLEL (saves 2-3 seconds!)
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                evidence_future = executor.submit(
                    self._get_evidence_block, query_english, top_k=2
                )
                cards_future = executor.submit(
                    cards_client.classify_claim, query_english
                )
                
                evidence_block, cites = evidence_future.result()
                is_misinformation, categories = cards_future.result()

            # Build appropriate prompt based on CARDS result
            if is_misinformation:
                # FALSE claim detected
                prompt = self._build_prompt_for_false_claim(
                    query_english, evidence_block, categories
                )
            else:
                # TRUE/NEUTRAL statement
                prompt = self._build_prompt_for_true_statement(
                    query_english, evidence_block
                )
            
            llm_answer = llm_client.invoke(prompt, temperature=0.0)

        # 5) Translate back if needed
        if src_lang != "English":
            final_answer = llm_client.translate_language(llm_answer, src_lang)[1]
        else:
            final_answer = llm_answer

        # 6) Log and return
        MisclassificationLog.objects.create(
            user=request.user,
            user_input=request.data.get("text", ""),
            llm_output=final_answer,
            is_misinformation=is_misinformation,
            references="\n".join(cites),
        )

        return Response(
            {
                "llm_response": final_answer,
                "misinformation": int(is_misinformation),
                "references": cites,
            },
            status=status.HTTP_200_OK,
        )


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
