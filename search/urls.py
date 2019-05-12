"""movietf URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from . import views
from django.urls import path

urlpatterns = [
    path('home/',views.finalhome,name='search-finalhome'),
    path('about/',views.finalhome,name='search-finalhome'),
    path('search/', views.finalhome, name='search-finalhome'),
    path('classify/',views.classify,name='classify'),
    path('classifyoutput/',views.classifyoutput,name='classifyoutput'),
    path('recommendation/',views.recommendation,name='recommendation'),
    path('recommendationoutput/',views.recommendationoutput,name='recommendationoutput'),
    path('behindthescene/',views.behindthescene,name='behindthescene'),

    path('outputformat/',views.outputformat,name ='search-outputformat'),

    path('',views.finalhome,name='search-finalhome'),


]
