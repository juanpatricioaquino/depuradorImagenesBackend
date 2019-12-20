
from django.http import JsonResponse
from django.test import TestCase
from django.conf.urls import url

# Create your tests here.

from . import views

app_name = 'depurador'

urlpatterns = [
    # ex: /polls/
    url(r'^$', views.index, name='index')
    # # ex: /polls/5/
    # url('<int:question_id>/', views.detail, name='detail'),
    # # ex: /polls/5/results/
    # url('<int:question_id>/results/', views.results, name='results'),
    # # ex: /polls/5/vote/
    # url('<int:question_id>/vote/', views.vote, name='vote'),
]