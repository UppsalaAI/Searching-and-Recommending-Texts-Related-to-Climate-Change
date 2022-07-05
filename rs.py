import pickle
import os.path
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec
from scipy import sparse
import time

from training import preprocess_data, preprocess_data_light

def preprocess_str(search_str):
    search_str_prep = preprocess_data([search_str])
    return search_str_prep[0]

def preprocess_str_light(search_str):
    search_str_prep = preprocess_data_light([search_str])
    return search_str_prep[0]

def string_to_vec(search_str):
    with open(os.path.join('pickle_files/', 'dictionary.pickle'), 'rb') as f:
        dictionary = pickle.load(f)
    search_vec = []
    for term in dictionary:
        search_vec.append(search_str.count(term))
    return search_vec

def create_similarity_vec(matrix, search_vec):
    sim_vec = []
    # num_of_docs = len(matrix[0])
    search_vec = np.array(search_vec).reshape(1,len(search_vec))
    shape = matrix.get_shape()

    for i in range(shape[1]):
        doc_vec = matrix.getcol(i).toarray()
        cs = cosine_similarity(search_vec, doc_vec.reshape(1,len(doc_vec)))
        sim_vec.append(cs)
    return sim_vec

def create_similarity_vec_tfidf(matrix, search_vec):
    sim_vec = []
    search_vec = np.array(search_vec).reshape(1,len(search_vec))
    shape = matrix.get_shape()

    for i in range(shape[1]):
        doc_vec = matrix[:,i].toarray()
        cs = cosine_similarity(search_vec, doc_vec.reshape(1,len(doc_vec)))
        sim_vec.append(cs)
    
    return sim_vec

def create_similarity_vec_d2v(vectors, search_vec):
    sim_vec = []
    search_vec = np.array(search_vec).reshape(1,len(search_vec))

    for vec in vectors:
        cs = cosine_similarity(search_vec, np.array(vec).reshape(1,len(vec)))
        sim_vec.append(cs)

    return sim_vec

def tf_idf(search_vec, len_search_str, dt_matrix):
    tfidf_vec = []
    shape = dt_matrix.get_shape()
    num_of_docs = shape[1] # number of columns in the matrix
    tf = np.array(search_vec)/len_search_str
    for i, _ in enumerate(search_vec):
        if search_vec[i] != 0:
            # idf = num_of_docs/np.count_nonzero(dt_matrix[i])
            idf = num_of_docs/len(dt_matrix.getrow(i).nonzero()[0])
            tfidf_vec.append(tf[i]*idf)
        else:
            tfidf_vec.append(0.0)
    return tfidf_vec

def top_n_sim(n, sim_vec):
    top_n = []
    for i in range(n):
        index_max = np.argmax(sim_vec)
        top_n.append(index_max + 1) # to get the index in the text file
        sim_vec[index_max] = 0
    return(top_n)


if __name__ == '__main__':
    search_strs = []
    with open(os.path.join("dataset/", "input.txt"), encoding="utf8") as file:
        for line in file.readlines():
            text = line.strip()
            search_strs.append(text)
    search_strs_prep = []
    search_strs_prep_light = []
    for search_str in search_strs:
        search_strs_prep.append(preprocess_str(search_str))
        search_strs_prep_light.append(preprocess_str_light(search_str))


    with open(os.path.join('pickle_files/', 'bow_matrix.pickle'), 'rb') as f:
        dt_matrix = pickle.load(f)
    search_vecs = []
    for search_str_prep in search_strs_prep:
        search_vecs.append(string_to_vec(search_str_prep))
    sim_vecs = []
    for search_vec in search_vecs:
        sim_vecs.append(create_similarity_vec(dt_matrix, search_vec))
    bow_top5_list = []
    for sim_vec in sim_vecs:
        bow_top5_list.append(top_n_sim(5, sim_vec))
    print(bow_top5_list)


    with open(os.path.join('pickle_files/', 'tfidf_matrix.pickle'), 'rb') as f:
        tfidf_matrix = pickle.load(f)
    tfidf_vecs = []
    for search_vec in search_vecs:
        tfidf_vecs.append(tf_idf(search_vec, len(search_str_prep), dt_matrix))
    sim_vecs = []
    for tfidf_vec in tfidf_vecs:
        sim_vecs.append(create_similarity_vec_tfidf(tfidf_matrix, tfidf_vec))
    tfidf_top5_list = []
    for sim_vec in sim_vecs:
        tfidf_top5_list.append(top_n_sim(5, sim_vec))


    with open(os.path.join('pickle_files/', 'd2v_model.pickle'), 'rb') as f:
        d2v_model = pickle.load(f)
    with open(os.path.join('pickle_files/', 'd2v_vectors.pickle'), 'rb') as f:
        d2v_vectors = pickle.load(f)
    d2v_vecs = []
    for search_str_prep_light in search_strs_prep_light:
        d2v_vecs.append(d2v_model.infer_vector(search_str_prep_light))
    sim_vecs = []
    for d2v_vec in d2v_vecs:
        sim_vecs.append(create_similarity_vec_d2v(d2v_vectors, d2v_vec))
    d2v_top5_list = []
    for sim_vec in sim_vecs:
        d2v_top5_list.append(top_n_sim(5, sim_vec))