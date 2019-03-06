from django.shortcuts import render,redirect
from django.http import HttpResponse
from utils.whsh1 import *
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





def outputformat(request):
    if request.method == "POST":
        print('*'*50)
        print(request.POST)
        query = request.POST['query']
        result = movie_search(query)
        return HttpResponse(result.to_html())
        #return render("search/result.html", result.to_html())
    else:
        print("no post")

        return redirect('/')




def about(request):
    return render(request, 'search/about.html')

# Create your views here.
