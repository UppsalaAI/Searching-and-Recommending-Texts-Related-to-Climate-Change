import numpy as np
import math
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer 
import nltk
import json
import os.path
from gensim import corpora
from gensim.models import LsiModel, TfidfModel, Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time
import pickle
from scipy import sparse
import itertools

# fetch the data from the specified file
def fetch_data(path, file_name):
    documents_list = []
    with open(os.path.join(path, file_name), encoding="utf8") as file:
        for line in file.readlines():
            text = line.strip()
            documents_list.append(text)
    return documents_list

# extract the POS-tag related to a word and return it
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    # TODO: this else is not completely legit, but it works for now
    else:
        return wordnet.ADV

def preprocess_data(data):
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()
    corpus = []
    for document in data:
        # tokenize text, remove punktuations
        tokens = regexp_tokenize(document, pattern='\w+')
        # remove stop words and make everything lower case
        tokens = [t.lower() for t in tokens if not t in stop_words]
        # Part-of-speach tagging
        tokens = nltk.pos_tag(tokens)
        # lemmatization, with POS-tag as input
        tokens = [lemmatizer.lemmatize(t[0], pos=get_wordnet_pos(t[1])) for t in tokens]
        # stemming
        tokens = [stemmer.stem(t) for t in tokens]
        corpus.append(tokens)
    return corpus

def preprocess_data_light(data):
    corpus = []
    for document in data:
        # tokenize text, remove punktuations
        tokens = regexp_tokenize(document, pattern='\w+')
        # make everything lower case
        tokens = [t.lower() for t in tokens]
        corpus.append(tokens)
    return corpus

def create_dt_matrix(corpus):
    # create dictionary
    dictionary = list(itertools.chain.from_iterable(corpus))
    dictionary = list(dict.fromkeys(dictionary))
    
    # create dt_matrix
    row_ind = np.array([], dtype=np.int8)
    col_ind = np.array([], dtype=np.int8)
    data = np.array([], dtype=np.int8)   

    for r, term in enumerate(dictionary):
        for c, doc in enumerate(corpus):
            term_occur_in_doc = doc.count(term)
            if term_occur_in_doc:
                row_ind = np.append(row_ind, r)
                col_ind = np.append(col_ind, c)
                data = np.append(data, term_occur_in_doc)

    dt_matrix = sparse.coo_matrix((data, (row_ind, col_ind)))

    return dictionary, dt_matrix

def tfidf_transformation(dt_matrix, dictionary, corpus):
    dt_matrix = sparse.dok_matrix(dt_matrix, dtype=np.int8)
    shape = dt_matrix.get_shape()
    num_of_docs = shape[1] # number of columns in the matrix

    non_zero = dt_matrix.nonzero()
    non_zero_x = non_zero[0]
    non_zero_y = non_zero[1]

    for i in range(len(non_zero_x)):
        x = non_zero_x[i]
        y = non_zero_y[i]
        row = dt_matrix.getrow(x)
        # tf_value = number of times the term occurs in the document / length of the document
        tf_value = dt_matrix.get((x,y))/len(corpus[y])
        # idf_value = number of documents in the corpus / number of documents containing the term
        idf_value = num_of_docs/len(row.nonzero()[0])
        dt_matrix[x,y] = tf_value*idf_value

    return dt_matrix

def get_doc_vectors(corpus, len_of_dict):
    # tag documents
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    # create model
    model = Doc2Vec(documents, vector_size=len_of_dict, seed=1, window=2, min_count=1, workers=4)
    # train model
    model.train(documents, total_examples=model.corpus_count, epochs=10)
    # save the model to a pickle file
    with open(os.path.join('pickle_files/', 'd2v_model.pickle'), 'wb') as f:
        pickle.dump(model, f)
    # delete training data?
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    vectors = []
    # get vector representation of each document and put in a list
    for doc in corpus:
        vector = model.infer_vector(doc)
        vectors.append(vector)
    return(vectors)

def bow(corpus):
    dictionary, dt_matrix = create_dt_matrix(corpus)
    # save the matrix and dictionary in pickle files
    with open(os.path.join('pickle_files/', 'bow_matrix.pickle'), 'wb') as f:
        pickle.dump(dt_matrix, f)
    with open(os.path.join('pickle_files/', 'dictionary.pickle'), 'wb') as f:
        pickle.dump(dictionary, f)
    return dictionary, dt_matrix

def tf_idf(corpus, dictionary, dt_matrix):
    tfidf_matrix = tfidf_transformation(dt_matrix, dictionary, corpus)
    # save the transformed matrix in a pickle file
    with open(os.path.join('pickle_files/', 'tfidf_matrix.pickle'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)

def doc2vec(corpus_light, len_of_dict):
    vectors = get_doc_vectors(corpus_light, len_of_dict)
    vectors = np.array(vectors)
    # save the vector embeddings in a pickle file
    with open(os.path.join('pickle_files/', 'd2v_vectors.pickle'), 'wb') as f:
        pickle.dump(vectors, f)


if __name__ == '__main__':
    # file_name = "abstracts.txt"
    file_name = "test.txt"
    documents = fetch_data("dataset/",file_name)
    corpus = preprocess_data(documents)
    corpus_light = preprocess_data_light(documents)

    # create the vector representations of the dataset using the bag of words method
    dictionary, dt_matrix = bow(corpus)
    
    # create the vector representations of the dataset using the tf-idf transformation on the corpus and dictionary created by the bag of words function
    tf_idf(corpus, dictionary, dt_matrix)

    # create the vector representations of the dataset using the doc2vec method
    len_of_dict = len(dictionary)
    doc2vec(corpus_light, len_of_dict)