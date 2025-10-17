from django.contrib.auth.models import User
from rest_framework import serializers
from rest_framework.exceptions import ValidationError as DRFValidationError
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.contrib.auth.hashers import make_password

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'password']
        
        extra_kwargs = {
            "password": {"write_only": True, "required": True},
        }
        

    def create(self, validated_data):
        try:
            validate_password(validated_data["password"])
        except ValidationError as e:
            raise DRFValidationError(e.messages)
        validated_data["password"] = make_password(validated_data["password"])
        return super().create(validated_data)