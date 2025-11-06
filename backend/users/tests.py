from rest_framework.test import APITestCase
from rest_framework import status
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token


class UserTests(APITestCase):
    def setUp(self):
        self.generate_token_url = "/api/users/generate-token/"
        self.user_admin = User.objects.create_superuser(
            username="admin",
            email="admin@example.com",
            password="password",
        )
        self.user_admin_token = Token.objects.get_or_create(user=self.user_admin)

    def test_post_generate_token(self):
        response = self.client.post(
            self.generate_token_url,
            data={
                "username": "admin",
                "password": "password",
            },
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["token"], self.user_admin_token[0].key)

    def test_post_generate_token_invalid_credentials(self):
        response = self.client.post(
            self.generate_token_url,
            data={
                "username": "admin",
                "password": "wrongpassword",
            },
        )
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_post_generate_token_missing_fields(self):
        response = self.client.post(
            self.generate_token_url,
            data={
                "username": "admin",
            },
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
