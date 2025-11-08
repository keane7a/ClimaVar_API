from rest_framework import permissions
from django.contrib.auth.models import User


class isAdminAndReadListOnly(permissions.BasePermission):
    """
    Custom permission to allow admins read a list only.
    """

    def has_permission(self, request, view):
        # Allow anyone to access 'list' action

        if view.action == "list" and request.user.is_staff:
            return True

        return False
