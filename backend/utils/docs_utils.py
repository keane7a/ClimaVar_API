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
class CheckMisclassificationRequestSerializer(serializers.Serializer):
    text = serializers.CharField(required=True)

class CheckMisclassificationResponseSerializer(serializers.Serializer):
    llm_response = serializers.CharField()
    misinformation = serializers.BooleanField()

class CheckMisclassificationRequestSerializer(serializers.Serializer):
    text = serializers.CharField(required=True)

CHECK_MISCLASSIFICATION_REQUEST = CheckMisclassificationRequestSerializer

CHECK_MISCLASSIFICATION_RESPONSES = {
    200: OpenApiResponse(response=CheckMisclassificationResponseSerializer),
    400: OpenApiResponse(
        response=ErrorResponseSerializer,
        examples=[
            OpenApiExample(
                "Invalid length",
                value={"message": "Input text must be between 10 and 300 characters."},
                response_only=True,
            ),
            OpenApiExample(
                "Unsupported language",
                value={"message": "Unsupported language for translation."},
                response_only=True,
            ),
            OpenApiExample(
                "Not climate related",
                value={
                    "message": "Query does not contain climate-related topics or sufficient climate-related keywords. Please rephrase your query to focus on climate-related content."
                },
                response_only=True,
            ),
        ],
    ),
}

CHECK_MISCLASSIFICATION_EXAMPLES = [
    OpenApiExample(
        "Check misclassification request",
        value={"text": "Climate change is a religion."},
        request_only=True,
    ),
    OpenApiExample(
        "Successful check response",
        value={
            "llm_response": "Offside! Claiming climate change is a religion is a total foul—it's backed by solid evidence, not just faith, and the science is clear, yet media often drops the ball by not linking extreme weather to climate change!\nReferences: Climate Change 2021: The Physical Science Basis, page number 2408 (2021) - https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_FullReport.pdf",
            "misinformation": True,
        },
        response_only=True,
    ),
]