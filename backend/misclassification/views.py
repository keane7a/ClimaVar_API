from http.client import HTTPException
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.permissions import IsAdminUser, AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import action

from misclassification.models import MisclassificationLog
from misclassification.serializer import MisclassificationLogSerializer

from misclassification.utils.utils import validate_input, cards_classify_claim, build_prompt, llm_answer, extract_final

class MisclassificationViewSet(viewsets.ModelViewSet):
    queryset = MisclassificationLog.objects.all()
    serializer_class = MisclassificationLogSerializer
    permission_classes = [IsAdminUser]


    @action(detail=False, methods=['post'], permission_classes=[IsAuthenticated], url_path='check-misclassification')
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
        
        # Init parameters
        prompt_version = "i0"   # choose among: i0,i1,i2,s0,s1,s2
        model = "gpt-4o-mini"
        temperature = 0.2
        
        text = request.data.get("text", "")
        
        # 1) Validate
        if not 10 < len(text) < 300:
            return Response({"error": "Input text must be between 10 and 300 characters."}, status=status.HTTP_400_BAD_REQUEST)


        # err = validate_input(text)
        # if err:
        #     return HTTPException(status_codxe=400, detail=err)

        # 2) CARDS classify
        cards = cards_classify_claim(text)

        # 3) Build prompt (and get metadata)
        prompt, family, uses_cot = build_prompt(text, cards, prompt_version)

        # 4) Generate
        raw = llm_answer(prompt, model=model, temperature=temperature)

        # 5) Extract

        # # 6) Log
        # log_run(
        #     user_text=user_text,
        #     cards_label=cards.label,
        #     cards_categories=cards.categories,
        #     cfg=cfg,
        #     prompt_family=family,
        #     uses_cot=uses_cot,
        #     final_text=final
        # )
        answer = extract_final(raw)
        # Add logging

        MisclassificationLog.objects.create(
            user = request.user,
            user_input = text,
            llm_output = answer,
            is_misinformation = cards.is_misinformation
        )
        
        return Response({"response": answer, "misinformation": cards.is_misinformation}, status=status.HTTP_200_OK)