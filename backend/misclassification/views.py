from http.client import HTTPException
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.permissions import IsAdminUser, AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import action
from drf_spectacular.utils import extend_schema_view, extend_schema
from misclassification.models import MisclassificationLog
from misclassification.serializer import MisclassificationLogSerializer
from misclassification.utils.utils import climate_keyword_score
from utils.docs_utils import (
    CHECK_MISCLASSIFICATION_RESPONSES,
    CHECK_MISCLASSIFICATION_EXAMPLES,
    CHECK_MISCLASSIFICATION_REQUEST,
)

from openai import OpenAI
from misclassification.utils.rag import LLMClient, CARDSClient
from misclassification.utils.embeddings import EmbeddingModel
from misclassification.utils.prompts import *
import chromadb
import re
import os


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

# print("Initialized LLM, CARDS, and Embedding models.")


@extend_schema_view(
    list=extend_schema(exclude=True),
    retrieve=extend_schema(exclude=True),
    create=extend_schema(exclude=True),
    update=extend_schema(exclude=True),
    partial_update=extend_schema(exclude=True),
    destroy=extend_schema(exclude=True),
)
class MisclassificationViewSet(viewsets.ModelViewSet):
    queryset = MisclassificationLog.objects.all()
    serializer_class = MisclassificationLogSerializer
    permission_classes = [IsAdminUser]

    def _get_evidence_block(self, query, top_k=3):
        """
        Get evidence block from chromadb based on query.

        Args:
            query (str): user query
            top_k (int): number of top results to retrieve
        Returns:
            evidence_block (str): formatted evidence block
            cites (list): list of citations as tuples (title, year, url)
        """

        max_char = 450
        results = embedding_model.query_chromadb(query, top_k)
        if not results:
            return "- (no evidence available)"

        # build evidence block and citations
        lines, cites = [], []
        for r in results:
            # Parse text
            text = r.get("text")
            text = text.strip()
            text = re.sub(r"\s+", " ", text)
            text = text[:max_char]
            text += "...\n"
            lines.append(f"- {text}")

            # Parse citation
            title, year, url, chunk_id = (
                r.get("title", ""),
                r.get("year", ""),
                r.get("url", ""),
                r.get("chunk_id", ""),
            )
            if chunk_id:
                chunk_id = chunk_id.split("_")[0]
            cites.append(f"{title}, page number {chunk_id} ({year}) - {url}")

        return "".join(lines), list(set(cites))

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
        Evaluate a user-supplied climate claim and return a concise fact-check.
        """

        """
        End-to-end:
        1) Validate length
        2) CARDS classify
        3) Build prompt (instruction or one-shot)
        4) LLM generation
        5) Extract FINAL (≤300 chars)
        6) Log with columns indicating prompt family and chain-of-thought
        """

        # Parameters
        top_k_evidence = 2

        query = request.data.get("text", "")

        # 1) Validate
        if not 10 < len(query) < 300:
            return Response(
                {"message": "Input text must be between 10 and 300 characters."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Translate to to english
        src_lang, query = llm_client.translate_language(query, "en")

        if not query:
            return Response(
                {"message": "Unsupported language for translation."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not climate_keyword_score(query):
            # Check using LLM to make sure.
            # Check for climate-related keywords
            classify_claim = llm_client.invoke(
                PROMPT_CLIMATE_TEXT_CLASSIFICATION.replace("{user_question}", query)
            )

            if classify_claim.strip() != "1":
                return Response(
                    {
                        "message": "Query does not contain climate-related topics or sufficient climate-related keywords. Please rephrase your query to focus on climate-related content."
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        # Determine if statement or question
        is_statement = llm_client.get_text_type(query)

        # query is a question
        if not is_statement:
            # print("it is a question")
            # Get evidence block
            evidence_block, cites = self._get_evidence_block(
                query, top_k=top_k_evidence
            )

            # print(evidence_block)
            # Get LLM answer based on embedding
            prompt = PROMPT_QUESTION.replace("{user_question}", query).replace(
                "{evidence_block}", evidence_block
            )

            is_misinformation = False

        else:  # Query is a statement
            # Classify statement
            is_misinformation, categories = cards_client.classify_claim(query)

            if is_misinformation:
                # print("it is a misinformation")
                # Convert user's query to neutral question for obtaining evidence block
                prompt = PROMPT_CONVERT_TO_NEUTRAL_QUESTION.replace(
                    "{user_question}", query
                )
                neutral_question = llm_client.invoke(prompt)

                # print("neutral question", neutral_question)
                # print("categories", categories)

                # Get evidence block
                evidence_block, cites = self._get_evidence_block(
                    neutral_question, top_k=top_k_evidence
                )

                # print("evidence block", evidence_block)

                # Get LLM answer based on embedding
                prompt = (
                    PROMPT_FALSE_CLAIM.replace("{user_claim}", query)
                    .replace("{evidence_block}", evidence_block)
                    .replace("{categories_summary}", categories)
                )

            else:  # if not misinformation
                # print("it is not a misinformation")
                evidence_block, cites = self._get_evidence_block(
                    query, top_k=top_k_evidence
                )
                prompt = PROMPT_QUESTION.replace("{user_question}", query).replace(
                    "{evidence_block}", evidence_block
                )

        # Get LLM answer
        llm_answer = llm_client.invoke(prompt)

        # Translate back
        final_answer = llm_client.translate_language(llm_answer, src_lang)[1]

        MisclassificationLog.objects.create(
            user=request.user,
            user_input=request.data.get("text", ""),
            llm_output=final_answer,
            is_misinformation=is_misinformation,
        )
        print(cites, "here")
        return Response(
            {
                "llm_response": final_answer,
                "misinformation": int(is_misinformation),
                "references": cites,
            },
            status=status.HTTP_200_OK,
        )
