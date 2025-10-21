from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin

# Create your models here.


# class User(AbstractBaseUser):
#     username = models.CharField(max_length=255, unique=True)
#     email = models.EmailField(max_length=255, unique=True)
#     first_name = models.CharField(max_length=255, blank=True, null=True)
#     last_name = models.CharField(max_length=255, blank=True, null=True)
#     USERNAME_FIELD = 'username'
