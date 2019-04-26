# SearchEngine
This **search engine** is a part of  movie recommendation system.

## Introduction to TF-IDF
TF-IDF stands for “Term Frequency — Inverse Data Frequency”. 
### Term Frequency (tf):
   TF gives us the frequency of the word in each document in the corpus. It is the ratio of number of times the word appears in a document compared to the total number of words in that document. It increases as the number of occurrences of that word within the document increases. Each document has its own tf.

![](/images/tf.png)


### Inverse Data Frequency (idf):
IDF used to calculate the weight of rare words across all documents in the corpus. The words that occur rarely in the corpus have a high IDF score. It is given by the equation below.


![](/images/idf.png)

IF_IDF:
Combining these two we come up with the TF-IDF score (w) for a word in a document in the corpus. It is the product of tf and idf:


![](/images/tfidf.png)


### Walkthrough the code
The preprocess function converts the text to lowercase and removes all the stop words from the text data.
    
    def preprocess(text):
    stop_words = stopwords.words('english')
    text = str(text)
    text = text.lower()
    #removing noise from data
    text = re.sub('[^a-z\s]', '', text.lower())
    text = [w for w in text.split() if w not in set(stop_words)]
    return ' '.join(text)
Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes. It reduce inflectional forms and sometimes derivationally related forms of a word to a common base form. I have used snowball stemmer as the movie data contains movies of different languages.   
    
    def stem(text):
    english_stemmer = SnowballStemmer('english')
    analyzer = CountVectorizer().build_analyzer()
    return (english_stemmer.stem(w) for w in analyzer(text))
    
Now that we have done with preprocessing of the data. Now we need to form the tf-idf matrix for the train data and test data.   And then we compute the cosine similarity. We output the top three results which aremost similar to the query string.
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

    
![](classify2.png)




REFERENCE:
https://medium.freecodecamp.org/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3
https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
