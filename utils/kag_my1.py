
import pandas as pd
from nltk.stem import WordNetLemmatizer
'''
import nltk
'''
import numpy as np

##hgjhg
'''
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
'''
from nltk import word_tokenize
from nltk import pos_tag
'''
from nltk.corpus import wordnet as wn
from sklearn.pipeline import Pipeline
'''
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
'''
from imblearn.over_sampling import SMOTE
import django
'''
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
'''
from sklearn.base import clone
'''
from sklearn.externals import joblib

def clean_sentence(sentence):
    sentence = sentence.lower()
    wordnet_lemmatizer = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', }

    new_sentence = []
    words = word_tokenize(sentence)
    role = pos_tag(words)
    for i, word in enumerate(words):
        if role[i][1] in VERB_CODES:
            new_word = wordnet_lemmatizer.lemmatize(word, 'v')
        else:
            new_word = wordnet_lemmatizer.lemmatize(word)
        if new_word not in stop and new_word.isalpha():
            new_sentence.append(new_word)


    s = ' '.join(new_sentence)

    return s

def trii(movie_plot):
    df = pd.read_csv("pavi.csv",nrows = 4500)
    df = df[['id', 'overview', 'genre']]
    df['overview'] = df['overview'].fillna(" ")
    df = df.dropna(axis=0, subset=['overview'])

    print("df", df.head(5))
    print("sum", df['overview'].isnull().sum())
    print("sum genre", df['genre'].isnull().sum())
    #df['overview_clean'] = df['overview'].apply(clean_sentence)
    #joblib.dump(df['overview_clean'], 'utils/overview_clean.pkl')
    df['overview_clean'] = joblib.load('utils/overview_clean.pkl')
    print(df['genre'].value_counts())
    genres = df['genre'].value_counts().reset_index()['index']

    probability_genre = round(df['genre'].value_counts() / len(df), 2)
    vectorizer = {'tfidf': TfidfVectorizer()}

    classifier = {'multinomial_nb': MultinomialNB()}
    n_vec = len(vectorizer)
    n_clf = len(classifier)





    pipe_dict = {}
    '''
    for genre in genres[:9]:
        df['genre_y'] = [1 if y == genre else 0 for y in df['genre']]

        vect_name = "tfidf"
        clf_name = "multinomial_nb"
        sampling_name = "over"
        vect = clone(vectorizer[vect_name])
        clf = clone(classifier[clf_name])


        x_vect = vect.fit_transform(df.overview_clean)


        x_vect, y = SMOTE().fit_sample(x_vect, df.genre_y)
        clf.fit(x_vect, y)

        #joblib.dump(clf, 'utils/classifier.pkl')

        #clf = joblib.load("utils/classifier.pkl")

        pipe_dict[genre] = Pipeline([('vect', vect), ('clf', clf)])
        
    joblib.dump(pipe_dict, 'utils/pipe_dict.pkl')
    '''
    pipe_dict = joblib.load('utils/pipe_dict.pkl')

    data = predict_genre(movie_plot, pipe_dict,probability_genre)
    return data
#predict_genre(movie_plot,Pipeline([('vect', 'tfidf'), ('clf', clf)]))
def predict_genre(s, pipe_dict,probability_genre):
    s_new = clean_sentence(s)
    genre_analyzed = []
    proba = []
    for genre, pipe in pipe_dict.items():
        res = pipe.predict_proba([s_new])
        print("res",res)
        genre_analyzed.append(genre)
        proba.append(res[0][1])
        print(proba)
    ###for now
    likelihood_res = np.divide(proba, probability_genre[:9])
    return genre_analyzed,proba,likelihood_res
