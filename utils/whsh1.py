

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

def preprocess(text):
    stop_words = stopwords.words('english')
    text = str(text)
    text = text.lower()
    #removing noise from data
    text = re.sub('[^a-z\s]', '', text.lower())
    text = [w for w in text.split() if w not in set(stop_words)]
    return ' '.join(text)


def stem(text):
    #im using a snowball stemmer here as there are movie data for different languages
    english_stemmer = SnowballStemmer('english')
    analyzer = CountVectorizer().build_analyzer()
    return (english_stemmer.stem(w) for w in analyzer(text))


def movie_search(query):

    df = pandas.read_csv("movies_data1.csv")
    df = df[['original_title', 'overview']]
    df['overview'] = df['overview'].fillna(" ")
    query = preprocess(query)
    count = CountVectorizer(analyzer=stem)
    moviedata_matrix = count.fit_transform(df['overview'])
    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(moviedata_matrix)
    test_matrix = count.transform([query])
    test_tfidf = tfidf_transformer.transform(test_matrix)
    cosine_similarity_score = cosine_similarity(test_tfidf, train_tfidf)
    result = np.argsort(cosine_similarity_score).tolist()
    df1 = df.iloc[result[0][-3:]]
    return df1.original_title.tolist(),df1.overview.tolist(),result[0][-3:]




print("--- %s seconds ---" % (time.time() - start_time))

