import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
movies_df = pd.read_csv('Mov_project data.csv')
#print(movies_df.head())
import tensorflow as tf
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
nltk.download('punkt')
def tokenize_and_stem(text):
    stemmer = SnowballStemmer('english')
    tokens = [words for sent in nltk.sent_tokenize(text) for words in nltk.word_tokenize(sent)]
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    stem = [stemmer.stem(words) for words in filtered_tokens]
    return stem
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words=None,
                                 use_idf=True, tokenizer=tokenize_and_stem,
                                 ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(x for x in movies_df['Plot'])
from sklearn.metrics.pairwise import cosine_similarity
similarity_distance = 1-cosine_similarity(tfidf_matrix)
def recommendmovies(title):
    ind = movies_df.loc[movies_df['Title']==title].index[0]
    mov_indices = np.argsort(similarity_distance)[1][:5]
    for i in mov_indices:
        print(movies_df.loc[i]['Title'])

recommendmovies('The Great Train Robbery')

