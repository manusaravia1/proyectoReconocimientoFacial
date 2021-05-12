from yolo import views
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin', admin.site.urls),
    path('video/<path:ip>', views.video, name='video'),
    path('', views.home, name='home'),
    path('ip', views.ip, name='ip'),
    path('upload', views.upload, name='upload'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
