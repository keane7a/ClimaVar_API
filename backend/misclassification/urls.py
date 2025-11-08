from rest_framework.routers import SimpleRouter
from misclassification.views import MisclassificationViewSet, MisclassificationLogViewSet


router = SimpleRouter()
router.register(
    r"misclassifications", MisclassificationViewSet, basename="misclassifications", 
)
router.register(   
    r"misclassification-logs", MisclassificationLogViewSet, basename="misclassification-logs"
)
