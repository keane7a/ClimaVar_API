from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate

# Create your views here.
class UserViewSet(viewsets.ViewSet):
    
    def generate_token(self, request):
        username = request.data.get("username", None)
        password = request.data.get("password", None)
        if username is None or password is None:
            return Response(
                {"message": "Username and password cannot be empty"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        user = authenticate(username=username, password=password)
        if user is None:
            return Response(
                {"message": "Invalid username or password!!"},
                status=status.HTTP_403_FORBIDDEN,
            )
        
        token = Token.objects.get_or_create(user=user)
        print(token)
        return Response(
            {"token": token[0].key},
            status=status.HTTP_200_OK,
        )
