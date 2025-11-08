from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action
from rest_framework.permissions import IsAdminUser, AllowAny
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from users.serializers import UserSerializer, LoginSerializer
from drf_spectacular.utils import extend_schema_view, extend_schema
from utils.docs_utils import (
    GenerateTokenSerializer,
    GENERATE_TOKEN_RESPONSES,
    GENERATE_TOKEN_EXAMPLES,
    LOGIN_RESPONSES, 
    LOGIN_EXAMPLES
)


# Create your views here.
@extend_schema_view(
    list=extend_schema(exclude=True),
    retrieve=extend_schema(exclude=True),
    create=extend_schema(exclude=True),
    update=extend_schema(exclude=True),
    partial_update=extend_schema(exclude=True),
    destroy=extend_schema(exclude=True),
)
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAdminUser]

    @extend_schema(
        summary="Generate Auth Token",
        description="Generate an authentication token for a user. To be used for check misclassification API access.",
        request=GenerateTokenSerializer,
        responses=GENERATE_TOKEN_RESPONSES,
        examples=GENERATE_TOKEN_EXAMPLES,
    )
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

    
    @extend_schema(
        summary="User Login",
        description="Log in a user and generate an authentication token.",
        request=LoginSerializer,
        responses=LOGIN_RESPONSES,
        examples=LOGIN_EXAMPLES,
    )
    @action(
        detail=False,
        methods=["post"],
        url_path="login",
        name="login",
        permission_classes=[AllowAny],
    )
    def login(self, request, *args, **kwargs): 
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
        if user.is_active:
            login(request, user)
            loginSerializer = LoginSerializer(
                {"message": "Logged in successfully", "user": user}
            ).data
            return Response(
                loginSerializer,
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                {"message": "This account is not active!!"},
                status=status.HTTP_401_UNAUTHORIZED,
            )