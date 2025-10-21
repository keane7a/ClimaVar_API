from rest_framework import serializers
from misclassification.models import MisclassificationLog


class MisclassificationLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = MisclassificationLog
        fields = ["id", "user", "user_input", "llm_output", "timestamp"]

        extra_kwargs = {
            "id": {"read_only": True},
            "timestamp": {"read_only": True},
            "user_input": {"required": True},
            "llm_output": {"required": True},
        }
