from rest_framework.routers import SimpleRouter
from misclassification.views import MisclassificationViewSet


router = SimpleRouter()
router.register(
    r"misclassifications", MisclassificationViewSet, basename="misclassifications"
)
