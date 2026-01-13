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
from misclassification.utils.prompts import *
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
        Reduced to top_k=2 and max_char=350 for faster performance.
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
        """Quick check if text is likely English to skip translation."""
        if not text:
            return True
        non_ascii = sum(1 for char in text if ord(char) > 127)
        return non_ascii / len(text) < 0.15

    def _check_climate_relevance(self, query):
        """
        Improved climate relevance check.
        More lenient to avoid false negatives.
        """
        prompt = f"""Determine if this text is related to climate science, environmental science, or climate change.

Text: "{query}"

Climate-related topics include:
- Climate change, global warming, greenhouse gases, carbon
- Temperature, weather, sea levels, ice, oceans
- Renewable energy, fossil fuels (in climate context)
- Deforestation, biodiversity, ecosystems
- Climate policy, IPCC, COP, Paris Agreement
- Environmental impacts of human activities

Answer ONLY with "1" if climate-related, or "0" if not.

Output:"""

        response = llm_client.invoke(prompt, temperature=0.0).strip()
        # Look for "1" in the response
        return "1" in response

    def _improve_prompt_for_misinformation(self, user_query, evidence_block, categories):
        """
        Build prompt for FALSE claims using football referee metaphors.
        Ensures consistency: use ONE strong call (RED CARD or OFFSIDE, not both).
        """
        prompt = f"""You are a climate expert refuting a FALSE claim like a football referee.

Rules:
- Your style is light, informal, and uses football referee language.
- Start with ONE strong referee call: "RED CARD!", "OFFSIDE!", or "FOUL!"
- Use the evidence to explain why the claim is wrong.
- Keep it to ONE sentence, max 280 characters.
- Be direct but friendly.

### EVIDENCE
{evidence_block}

### INPUT (FALSE CLAIM)
"{user_query}"
Misinformation categories: {categories}

Output ONLY (no preamble):
<one-sentence refutation using referee language, ≤280 chars>"""
        
        return prompt

    def _improve_prompt_for_question(self, user_question, evidence_block):
        """
        Build prompt for QUESTIONS using football referee metaphors.
        Uses positive calls like GOAL!, PLAY ON!, etc.
        """
        prompt = f"""You are a climate expert answering a question like a football referee.

Rules:
- Your style is light, informal, and uses football referee language.
- Use positive referee calls: "GOAL!", "PLAY ON!", "FAIR PLAY!", "VAR CONFIRMS!"
- Use the evidence to answer the question accurately.
- Keep it to ONE sentence, max 280 characters.
- Be enthusiastic and clear.

### EVIDENCE
{evidence_block}

### QUESTION
"{user_question}"

Output ONLY (no preamble):
<one-sentence answer using referee language, ≤280 chars>"""
        
        return prompt

    def _improve_prompt_for_true_statement(self, user_statement, evidence_block):
        """
        Build prompt for TRUE statements using football referee metaphors.
        Uses confirming calls.
        """
        prompt = f"""You are a climate expert confirming an accurate statement like a football referee.

Rules:
- Your style is light, informal, and uses football referee language.
- Use confirming referee calls: "GOAL!", "PLAY ON!", "VAR CONFIRMS!", "FAIR PLAY!"
- Use the evidence to confirm and add context.
- Keep it to ONE sentence, max 280 characters.
- Be supportive of accurate information.

### EVIDENCE
{evidence_block}

### STATEMENT (ACCURATE)
"{user_statement}"

Output ONLY (no preamble):
<one-sentence confirmation using referee language, ≤280 chars>"""
        
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
        V2 Pipeline - Optimized for both quality and performance:
        
        QUESTIONS (contains "?"):
          1. Translate if needed (skip if English)
          2. Check climate relevance
          3. Get evidence (no CARDS needed for questions)
          4. Generate answer
        
        STATEMENTS (no "?"):
          1. Translate if needed (skip if English)
          2. Check climate relevance
          3. Run CARDS + Get evidence IN PARALLEL
          4. Generate appropriate response based on CARDS result
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

        # 3) Check climate relevance
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
            # QUESTION PATH - No normalization, no CARDS needed
            # Just get evidence and answer
            evidence_block, cites = self._get_evidence_block(query_english, top_k=2)
            
            # Build prompt for question
            prompt = self._improve_prompt_for_question(query_english, evidence_block)
            
            # Generate answer
            llm_answer = llm_client.invoke(prompt)
            
            is_misinformation = False
            categories = ""

        else:
            # STATEMENT PATH - Run CARDS to check for misinformation
            # Get evidence and CARDS IN PARALLEL (saves 2-3 seconds!)
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
                # FALSE claim - use refutation prompt
                prompt = self._improve_prompt_for_misinformation(
                    query_english, evidence_block, categories
                )
            else:
                # TRUE/NEUTRAL statement - use confirmation prompt
                prompt = self._improve_prompt_for_true_statement(
                    query_english, evidence_block
                )
            
            # Generate answer
            llm_answer = llm_client.invoke(prompt)

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
