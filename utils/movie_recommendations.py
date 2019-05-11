import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
df_credits=pd.read_csv('tmdb_5000_credits.csv')
df_movies=pd.read_csv('tmdb_5000_movies.csv')
df_credits.columns = ['id','tittle','cast','crew']
df_movies= df_movies.merge(df_credits,on='id')


tfidf = TfidfVectorizer(stop_words='english')
df_movies['overview'] = df_movies['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df_movies['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df_movies.index, index=df_movies['title']).drop_duplicates()

def recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df_movies['title'].iloc[movie_indices]

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df_movies[feature] = df_movies[feature].apply(literal_eval)

def extract_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

def get_production_company(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 1:
            names = names[:1]
        return names

df_movies['director'] = df_movies['crew'].apply(extract_director)
df_movies['production_companies'] = df_movies['production_companies'].apply(get_production_company)
features = ['cast', 'keywords', 'genres']
for feature in features:
    df_movies[feature] = df_movies[feature].apply(get_list)

df_movies[['title', 'cast', 'director', 'keywords', 'genres','production_companies']].head(3)

def data_clean(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
features = ['cast', 'keywords', 'director','production_companies','genres']

for feature in features:
    df_movies[feature] = df_movies[feature].apply(data_clean)


def form_metadata_string(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast'])  + ' ' + x['production_companies'] + ' ' +  x['director'] + ' '.join(x['genres'])
df_movies['metadata'] = df_movies.apply(form_metadata_string, axis=1)



count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df_movies['metadata'])
cosine_sim_y = cosine_similarity(count_matrix, count_matrix)
df_movies = df_movies.reset_index()
indices = pd.Series(df_movies.index, index=df_movies['title'])

def result(moviename):
    x = recommendations(moviename)
    y = recommendations(moviename, cosine_sim_y)
    print("x is ",x)
    print("y is ",y)
    return x.tolist(),y.tolist()

#x,y = result("The Dark Knight Rises")
#print("answer",x)