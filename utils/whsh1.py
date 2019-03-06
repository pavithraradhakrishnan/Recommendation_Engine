import os
import string

import numpy as np
import pandas
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import re
start_time = time.time()
#nltk.download('stopwords')

#print("shape of df",df.shape)
#print(df.head(5))
def preprocess(text):
    stop_words = stopwords.words('english')
    text = str(text)
    #print("text is ",type(text))
    text = text.lower()
    #removing noise from data
    text = re.sub('[^a-z\s]', '', text.lower())
    #j = list(string.punctuation)
    #stop_words = set(i).union(j)
    text = [w for w in text.split() if w not in set(stop_words)]
    return ' '.join(text)


def stem(text):
    #im using a snowball stemmer here as there are movie data for different languages
    english_stemmer = SnowballStemmer('english')
    analyzer = CountVectorizer().build_analyzer()
    return (english_stemmer.stem(w) for w in analyzer(text))



def movie_search(query):
    #df = pandas.read_csv("C:\Certificates\Ex_Files_TensorFlow\Ex_Files_TensorFlow\movietf\movies_metadata.csv")
    df = pandas.read_csv("movies_metadata.csv")
    df = df[['original_title', 'overview']]

    df['overview'] = df['overview'].fillna(" ")
    #print(df[['overview']].head(5))

    query = preprocess(query)

    count = CountVectorizer(analyzer=stem)

    moviedata_matrix = count.fit_transform(df['overview'])

    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(moviedata_matrix)
    #query = process_text(query)
    test_matrix = count.transform([query])
    test_tfidf = tfidf_transformer.transform(test_matrix)
    cosine_similarity_score = cosine_similarity(test_tfidf, train_tfidf)
    result = np.argsort(cosine_similarity_score).tolist()

    df1 = df.iloc[result[0][-10:]]
    return df1.iloc[::-1]

#movies = movie_search("sdffmxcgfdg")

print("--- %s seconds ---" % (time.time() - start_time))
#print(movies)




