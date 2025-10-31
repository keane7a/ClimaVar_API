from rest_framework import serializers
from drf_spectacular.utils import OpenApiExample, OpenApiResponse


class ErrorResponseSerializer(serializers.Serializer):
    message = serializers.CharField()

###############
# User Documentations
###############
class GenerateTokenSerializer(serializers.Serializer):
    username = serializers.CharField(required=True)
    password = serializers.CharField(required=True)
    
class GenerateTokenResponseSerializer(serializers.Serializer):
    token = serializers.CharField()



GENERATE_TOKEN_RESPONSES = {
    200: GenerateTokenResponseSerializer,
    400: OpenApiResponse(
    response=ErrorResponseSerializer,
    examples=[
            OpenApiExample(
                "Missing credentials",
                value={"message": "Username and password cannot be empty"},
                response_only=True,
            )
        ],
    ),
    403: OpenApiResponse(
        response=ErrorResponseSerializer,
        examples=[
            OpenApiExample(
                "Invalid credentials",
                value={"message": "Invalid username or password!!"},
                response_only=True,
            )
        ],
    ),
}

GENERATE_TOKEN_EXAMPLES = [
    OpenApiExample(
        "Successful token generation",
        value={"username": "john_doe", "password": "password123"},
        request_only=True,
    ),
    OpenApiExample(
        "Token response",
        value={"token": "xxxxx"},
        response_only=True,
    ),
]



###############
# Misclassification Documentations
###############
