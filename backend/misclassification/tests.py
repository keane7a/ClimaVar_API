from rest_framework.test import APITestCase
from rest_framework import status
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token


class MisclassificationTest(APITestCase):
    def setUp(self):
        self.check_misclassification_url = "/api/misclassifications/check-misclassification/"
        self.user = User.objects.create_user(
            username="testuser",
            email="admin@example.com",
            password="password"
        )
        self.user_token = Token.objects.get_or_create(user=self.user)[0].key
    
    
    def test_post_check_misclassification(self):
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.user_token)
        response = self.client.post(
            self.check_misclassification_url,
            data={
                "text": "Climate change is a religion."
            }
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        #self.assertEqual(1, response.data["misinformation"]) # True it is a misinformation
        self.assertIn("llm_response", response.data)
        
    def test_post_check_misclassification_character_range(self):
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.user_token)
        response = self.client.post(
            self.check_misclassification_url,
            data={
                "text": "Climate"
            }
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Input", response.data["message"])
    
    def test_post_check_misclassification_unsupported_traslation(self):
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.user_token)
        response = self.client.post(
            self.check_misclassification_url,
            data={
                "text": "Halo, anda siapa ya?",
            }
        )
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Unsupported", response.data["message"])

    def test_post_check_misclassification_not_climate_related(self):
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.user_token)
        response = self.client.post(
            self.check_misclassification_url,
            data={
                "text": "This is a house.",
            }
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Query", response.data["message"])