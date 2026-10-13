from rest_framework import serializers
from users.serializers import LoginSerializer
from drf_spectacular.utils import OpenApiExample, OpenApiResponse
from users.serializers import UserSerializer


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

LOGIN_RESPONSES = {
    200: LoginSerializer,
    403: OpenApiResponse(
        response=ErrorResponseSerializer,
        examples=[
            OpenApiExample(
                "Invalid username or passoword",
                value={"message": "Invalid username or password!!"},
                response_only=True,
            )
        ],
    ),
    401: OpenApiResponse(
        response=ErrorResponseSerializer,
        examples=[
            OpenApiExample(
                "Inactive account",
                value={"message": "This account is not active!!"},
                response_only=True,
            )
        ],
    ),
}

LOGIN_EXAMPLES = [
    OpenApiExample(
        "Successful login",
        value={"username": "john_doe", "password": "password123"},
        request_only=True,
    ),
    OpenApiExample(
        "Login response",
        value={"message": "Logged in successfully", "user": UserSerializer().data},
        response_only=True,
    ),
]


###############
# Misclassification Documentations
###############
class CheckMisclassificationRequestSerializer(serializers.Serializer):
    text = serializers.CharField(
        required=True,
        help_text="Climate claim or question (10-300 characters).",
    )
    language = serializers.ChoiceField(
        choices=["english", "portuguese", "spanish", "arabic"],
        required=False,
        default="english",
        help_text="Language of the input and of the response. Defaults to english.",
    )


class CheckMisclassificationResponseSerializer(serializers.Serializer):
    llm_response = serializers.CharField(
        help_text="Football-style verdict in the requested language."
    )
    misinformation = serializers.IntegerField(
        help_text="0 = accurate, 1 = misinformation, 2 = partial / needs context."
    )
    references = serializers.ListField(child=serializers.CharField())
    source = serializers.ChoiceField(
        choices=["climategpt", "backup"],
        help_text="climategpt = primary pipeline, backup = alternative pipeline.",
    )


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
                value={
                    "message": "Language 'french' is not supported yet. ClimaVAR currently supports: English, Portuguese, Spanish and Arabic."
                },
                response_only=True,
            ),
            OpenApiExample(
                "Not climate related",
                value={
                    "message": "Query does not contain climate-related topics. Please rephrase your query to focus on climate-related content."
                },
                response_only=True,
            ),
        ],
    ),
}

CHECK_MISCLASSIFICATION_EXAMPLES = [
    OpenApiExample(
        "Request (English)",
        value={"text": "Climate change is a religion.", "language": "english"},
        request_only=True,
    ),
    OpenApiExample(
        "Request (Arabic)",
        value={"text": "التغير المناخي ليس سببه النشاط البشري", "language": "arabic"},
        request_only=True,
    ),
    OpenApiExample(
        "Response (misinformation)",
        value={
            "llm_response": "RED CARD! Climate change is not a belief system: it is backed by decades of evidence and the scientific consensus of the IPCC.",
            "misinformation": 1,
            "references": [],
            "source": "climategpt",
        },
        response_only=True,
    ),
    OpenApiExample(
        "Response (accurate)",
        value={
            "llm_response": "GOAL! Sea levels are indeed rising due to climate change, driven by melting ice and the thermal expansion of warming oceans.",
            "misinformation": 0,
            "references": [],
            "source": "climategpt",
        },
        response_only=True,
    ),
]
