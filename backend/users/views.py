from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action
from rest_framework.permissions import IsAdminUser, AllowAny
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from users.serializers import UserSerializer


# Create your views here.
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAdminUser]

    @action(
        detail=False,
        methods=["post"],
        name="generate_token",
        url_path="generate-token",
        permission_classes=[AllowAny],
    )
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
        
        return Response(
            {"token": token[0].key},
            status=status.HTTP_200_OK,
        )
