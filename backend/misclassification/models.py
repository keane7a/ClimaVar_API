from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class MisclassificationLog(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    user_input = models.TextField(max_length=300)
    llm_output = models.TextField(max_length=300)
    is_misinformation = models.BooleanField(null=False)
    timestamp = models.DateTimeField(auto_now_add=True)
    references = models.TextField(null=True)
