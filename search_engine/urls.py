from django.urls import path
from . import views

app_name = 'search_engine'
urlpatterns = [
    path('', views.index, name='index'),
    path('<int:article_id>/', views.detail, name='detail'),
    path('search/str=<str:search_str>/', views.search, name='search'),
]
