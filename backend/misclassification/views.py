from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.permissions import IsAdminUser, AllowAny
from rest_framework.response import Response
from rest_framework.decorators import action

from misclassification.models import MisclassificationLog
from misclassification.serializer import MisclassificationLogSerializer
# Create your views here.

class MisclassificationViewSet(viewsets.ModelViewSet):
    queryset = MisclassificationLog.objects.all()
    serializer_class = MisclassificationLogSerializer
    permission_classes = [IsAdminUser]
    
    
    @action(detail=True, methods=['post'], permission_classes=[AllowAny], url_path='check-misclassification')
    def check_misclassification(self, request):
        
        return Response({"status": "Misclassification endpoint is working"}, status=status.HTTP_200_OK)