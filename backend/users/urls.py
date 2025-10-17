from rest_framework.routers import SimpleRouter

from users.views import UserViewSet

router = SimpleRouter()
router.register(r"users", UserViewSet, basename="users")