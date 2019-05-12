

from django.shortcuts import render,redirect
from django.http import HttpResponse


from utils.whsh1 import *
from utils.kag_my1 import *
from utils.movie_recommendations import *
'''
import matplotlib.pyplot as plt
import matplotlib
'''
from matplotlib import pylab
from pylab import *
import PIL, PIL.Image
import io
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
def behindthescene(request):
    return render(request,'search/behindthescene.html')

def classify(request):
    return render(request,'search/classify.html')
def recommendation(request):
    return render(request,'search/recommendation.html')
def classifyoutput(request):
    if request.method == "POST":
        print('*'*50)
        print(request.POST)
        query = request.POST['query']

        genre_analyzed,proba,likelihood = trii(query)
        data = pd.DataFrame({'genre': genre_analyzed, 'proba': proba,'likelihood':likelihood})
        data = data.sort_values(by='proba', ascending=True)
        ################ trial code starts#############################
        plt.subplot(2, 1, 1)
        plt.barh(data['genre'], data['proba'])
        plt.title('CLASSIFICATION RESULT ')
        plt.xlabel('genre')
        plt.ylabel('probability')


        plt.subplot(2, 1, 2)
        plt.barh(data['genre'], data['likelihood'])
        plt.title('LIKELIHOOD RESULT (for more calculations visit CLASSIFICATION REPORTS tab)')
        plt.xlabel('genre')
        plt.ylabel('likelihood')
        plt.tight_layout()



        #####################trial ends#################################

        '''
        correct code
        plt.subplot(1,1,1)
        ax = data.plot(x='genre', y='proba', kind='barh')


        plt.title('Classification result')
        ax.legend()

        plt.subplot(1,2,1)
        ax1= data.plot(x='genre', y='likelihood', kind='barh')

        plt.title('Likelihood result')
        ax1.legend()
        '''

        #print("data in html",data)
        '''
        plt.bar(genre, proba)
        plt.title('Classification result')
        '''

        #ax.legend()
        #plt.show()

        buffer = io.BytesIO()
        canvas = pylab.get_current_fig_manager().canvas
        canvas.draw()
        pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
        pilImage.save(buffer, "PNG")
        pylab.close()

        ################################################################################
        '''
        data2 = pd.DataFrame({'genre': genre_analyzed, 'likelihood': likelihood})
        # data2 = data2.sort_values(by='likelihoood', ascending=True)
        ax2 = data2.plot(x='genre', y='likelihood', kind='barh')
        plt.title('Likelihood result')
        ax2.legend()
        buffer = io.BytesIO()
        canvas = pylab.get_current_fig_manager().canvas
        canvas.draw()
        pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
        pilImage.save(buffer, "PNG")
        pylab.close()
        '''

        # Send buffer in a http response the the browser with the mime type image/png set
        return HttpResponse(buffer.getvalue(),content_type="image/png")
        #return render(request, 'search/classifyoutput.html', {'output': output})

        #return render(request,'search/classifyoutput.html')

        #return render(request,'search/classifyoutput.html', {'output':output})
        #return HttpResponse(plt.to_html())
    else:
        print("no post")

        return redirect('/')


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
