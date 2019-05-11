from django.shortcuts import render,redirect
from django.http import HttpResponse
from utils.whsh1 import *
from utils.movie_recommendations import *
#from django.contrib.auth.forms import se

from subprocess import Popen, PIPE, STDOUT
movies = [
    {
        'title':'Toy story',
        'overview':'Andy met ghfnh jhgjhg',

    },
    {
        'title':'Toy story',
        'overview':'Andy met ghfnh jhgjhg',
    }
]



def home(request):
    #return HttpResponse('<h1>Blog HOme </h1>')
    context = {
        'movies' : movies
    }
    #result = utils.get_search_results("birthday brings Buzz Lightyear onto the scene")
    #print(result)
    return render(request,'search/home.html',context)

def finalhome(request):
    return render(request,'search/finalhome.html')

def recommendationoutput(request):

    if request.method == "POST":
        print('*' * 50)
        print(request.POST)
        moviename = request.POST['query']
        x,y = result(moviename)
        return render(request, 'search/recommendationoutput.html', {'result': zip(x,y)})
    else:
        print("no post")

        return redirect('/')
def recommendation(request):
    return render(request,'search/recommendation.html')

def outputformat(request):

    if request.method == "POST":
        print('*' * 50)
        print(request.POST)
        query = request.POST['query']
        original_title,overview,cosine_scores = movie_search(query)
        return render(request, 'search/classify_scores_output.html', {'result': zip(original_title,overview,cosine_scores)})
    else:
        print("no post")
        return redirect('/')




def about(request):
    return render(request, 'search/about.html')

# Create your views here.
