import os

import numpy as np
import pandas
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

#nltk.download('stopwords')

#print("shape of df",df.shape)
#print(df.head(5))
def stemming(text):
    english_stemmer = SnowballStemmer('english')
    analyzer = CountVectorizer().build_analyzer()
    return (english_stemmer.stem(w) for w in analyzer(text))



def get_search_results(query):
    df = pandas.read_csv("C:\Certificates\Ex_Files_TensorFlow\Ex_Files_TensorFlow\movietf\movies_metadata.csv")
    df = df[['original_title', 'overview']]

    df['overview'] = df['overview'].fillna(" ")
    #print(df[['overview']].head(5))




    count = CountVectorizer(analyzer=stemming)

    count_matrix = count.fit_transform(df['overview'])

    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(count_matrix)
    #query = process_text(query)
    query_matrix = count.transform([query])
    query_tfidf = tfidf_transformer.transform(query_matrix)
    sim_score = cosine_similarity(query_tfidf, train_tfidf)
    sorted_indexes = np.argsort(sim_score).tolist()
    return df.iloc[sorted_indexes[0][-1:]]

movies = get_search_results("birthday brings Buzz Lightyear onto the scene")
print(movies)




