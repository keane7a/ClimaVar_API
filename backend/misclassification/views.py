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

        max_char = 350  # Reduced from 450 for faster processing
        results = embedding_model.query_chromadb(query, top_k)
        if not results:
            return "- (no evidence available)", []

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
            cites.append(f"{title}, ({year}) - {url}")

        return "".join(lines), list(set(cites))

    def _is_likely_english(self, text):
        """
        Quick heuristic to detect if text is likely English.
        Skips translation step for English text, saving 1-2 seconds.
        """
        non_ascii = sum(1 for char in text if ord(char) > 127)
        if len(text) == 0:
            return True
        return non_ascii / len(text) < 0.15  # Less than 15% non-ASCII

    def _normalize_to_claim(self, query):
        """
        Convert any input (question, statement, negative) to a neutral claim.
        This makes downstream processing consistent and improves LLM quality.
        
        Examples:
        "Is global warming real?" → "Global warming is real"
        "Climate change is a hoax" → "Climate change is a hoax" (unchanged)
        "I don't think CO2 affects temperature" → "CO2 does not affect temperature"
        """
        
        prompt = f"""Convert this input to a simple factual claim statement.

Input: "{query}"

Rules:
- If it's a question, convert to a statement
- If it's already a statement, keep it as is
- Preserve the meaning and any negations
- Keep it concise (one sentence)
- Do not add judgment or commentary

Output only the normalized claim, nothing else.

Examples:
Input: "Is global warming real?"
Output: Global warming is real

Input: "Climate change is fake"
Output: Climate change is fake

Input: "Does CO2 cause warming?"
Output: CO2 causes warming

Input: "I don't believe sea levels are rising"
Output: Sea levels are not rising

Now normalize this input:
Input: "{query}"
Output:"""

        normalized = llm_client.invoke(prompt, temperature=0.0).strip()
        # Remove any quotes that might be added
        normalized = normalized.strip('"').strip("'")
        return normalized

    def _check_climate_relevance(self, query):
        """
        Improved climate relevance check using LLM with explicit criteria.
        More robust than simple keyword matching.
        """
        
        prompt = f"""You are a precise text classifier for climate-related content.

Text: "{query}"

Climate-related topics include:
- Climate change, global warming, greenhouse gases
- Carbon emissions, CO2, methane, fossil fuels
- Temperature changes, sea level rise, extreme weather
- Renewable energy, solar, wind (in climate/environment context)
- Deforestation, biodiversity loss, ecosystems
- IPCC, Paris Agreement, COP conferences, climate policy
- Ice melting, glaciers, polar regions, Arctic/Antarctic
- Ocean acidification, coral bleaching
- Climate science, climate models, climate data
- Environmental impacts of human activities

NOT climate-related:
- Pure financial topics (unless directly about climate finance)
- General geography without climate connection
- Spam or gibberish
- Non-climate environmental topics (e.g., plastic pollution without climate link)

Reasoning: [Think about whether this is genuinely about climate science]
Output: [1 if climate-related, 0 if not]

Format your response as:
Reasoning: <your reasoning>
Output: <0 or 1>"""

        response = llm_client.invoke(prompt, temperature=0.0)
        # Extract the output number
        return "1" in response[-10:]  # Check last 10 chars for "Output: 1"

    def _build_unified_prompt(self, original_query, normalized_query, evidence_block, 
                              is_misinformation, categories, is_question):
        """
        Build a unified prompt that works for all cases.
        Uses football referee metaphors consistently.
        """
        
        # Determine the appropriate prompt template
        if is_misinformation:
            # Use the existing FALSE_CLAIM template structure but with unified approach
            prompt = f"""You are a climate expert refuting a claim like a friendly football referee.

Rules:
- Your style is light, informal, and full of football referee lingo.
- Call out the misinformation. You MUST start with a referee call like "RED CARD!", "OFFSIDE!", "FOUL!", or "VAR SAYS NO!".
- Use the evidence snippets to correct the misinformation.
- Keep it to ONE sentence, max 300 characters.
- Be direct but friendly - you're helping people understand the truth.

### EVIDENCE SNIPPETS
{evidence_block}

### INPUT
Original: {original_query}
Claim being checked: {normalized_query}
Misinformation detected: {categories}

Output ONLY (no preamble):
<one-sentence refutation, ≤300 chars, informal football referee commentary>"""
        
        elif is_question:
            # Question format - use existing QUESTION template style
            prompt = f"""You are a climate expert answering a question like a friendly football referee.

Rules:
- Your style is light, informal, and full of football referee lingo.
- Use positive referee calls like "GOAL!", "PLAY ON!", "FAIR PLAY!", "VAR CONFIRMS!" when appropriate.
- Use the evidence snippets to answer the user's question accurately.
- Keep it to ONE sentence, max 300 characters.
- Be enthusiastic and clear.

### EVIDENCE SNIPPETS
{evidence_block}

### INPUT
Question: {original_query}

Output ONLY (no preamble):
<one-sentence answer, ≤300 chars, informal football referee commentary>"""
        
        else:
            # Statement that's not misinformation - neutral/confirming
            prompt = f"""You are a climate expert evaluating a statement like a friendly football referee.

Rules:
- Your style is light, informal, and full of football referee lingo.
- Use affirmative referee calls like "GOAL STANDS!", "PLAY ON!", "VAR CONFIRMS!", "FAIR PLAY!".
- Use the evidence snippets to confirm or provide nuance to the statement.
- Keep it to ONE sentence, max 300 characters.
- Be supportive of accurate information.

### EVIDENCE SNIPPETS
{evidence_block}

### INPUT
Statement: {original_query}
Checking: {normalized_query}

Output ONLY (no preamble):
<one-sentence evaluation, ≤300 chars, informal football referee commentary>"""

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
        Simplified, optimized pipeline:
        1. Validate & translate (with smart English detection)
        2. Normalize input to standard format
        3. Check climate relevance
        4. Get evidence + CARDS classification in parallel
        5. Generate unified response
        """

        query = request.data.get("text", "")

        # 1) Validate length
        if not 10 < len(query) < 300:
            return Response(
                {"message": "Input text must be between 10 and 300 characters."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 2) Smart translation - skip if already English (saves 1-2 seconds)
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

        # 3) Check climate relevance (improved prompt)
        is_climate_related = self._check_climate_relevance(query_english)
        
        if not is_climate_related:
            return Response(
                {
                    "message": "Query does not contain climate-related topics or sufficient climate-related keywords. Please rephrase your query to focus on climate-related content."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 4) Determine if it's a question (for prompt selection)
        is_question = "?" in query_english
        
        # 5) Normalize input to claim format for consistency
        normalized_query = self._normalize_to_claim(query_english)

        # 6) Get evidence and CARDS classification IN PARALLEL (saves 2-3 seconds!)
        with ThreadPoolExecutor(max_workers=2) as executor:
            evidence_future = executor.submit(
                self._get_evidence_block, normalized_query, top_k=2
            )
            cards_future = executor.submit(
                cards_client.classify_claim, normalized_query
            )
            
            evidence_block, cites = evidence_future.result()
            is_misinformation, categories = cards_future.result()

        # 7) Build unified prompt
        prompt = self._build_unified_prompt(
            original_query=query_english,
            normalized_query=normalized_query,
            evidence_block=evidence_block,
            is_misinformation=is_misinformation,
            categories=categories,
            is_question=is_question
        )

        # 8) Generate answer
        llm_answer = llm_client.invoke(prompt)

        # 9) Translate back (if needed)
        if src_lang != "English":
            final_answer = llm_client.translate_language(llm_answer, src_lang)[1]
        else:
            final_answer = llm_answer

        # 10) Log and return
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
