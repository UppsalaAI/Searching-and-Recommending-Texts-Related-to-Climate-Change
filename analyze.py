import os.path
import pickle

if __name__ == '__main__':
    with open(os.path.join('pickle_files/', 'bow_top5_list.pickle'), 'rb') as f:
        bow_top5_list = pickle.load(f)
    with open(os.path.join('pickle_files/', 'tfidf_top5_list.pickle'), 'rb') as f:
        tfidf_top5_list = pickle.load(f)
    with open(os.path.join('pickle_files/', 'd2v_top5_list.pickle'), 'rb') as f:
        d2v_top5_list = pickle.load(f)

    bow_bool_original = []
    bow_bool_modified = []
    tfidf_bool_original = []
    tfidf_bool_modified = []
    d2v_bool_original = []
    d2v_bool_modified = []

    # for i in range(0, 35):
    for i in range(0, 20):
        if i+16 in bow_top5_list[i]:
            bow_bool_original.append(True)
        else:
            bow_bool_original.append(False)
    # for i in range(35, 70):
    for i in range(20, 40):
        # if i-34 in bow_top5_list[i]:
        if i-4 in bow_top5_list[i]:
            bow_bool_modified.append(True)
        else:
            bow_bool_modified.append(False)

    # for i in range(0, 35):
    for i in range(0, 20):
        if i+16 in tfidf_top5_list[i]:
            tfidf_bool_original.append(True)
        else:
            tfidf_bool_original.append(False)
    # for i in range(35, 70):
    for i in range(20, 40):
        # if i-34 in tfidf_top5_list[i]:
        if i-4 in tfidf_top5_list[i]:
            tfidf_bool_modified.append(True)
        else:
            tfidf_bool_modified.append(False)

    # for i in range(0, 35):
    for i in range(0, 20):
        if i+16 in d2v_top5_list[i]:
            d2v_bool_original.append(True)
        else:
            d2v_bool_original.append(False)
    # for i in range(35, 70):
    for i in range(20, 40):
        # if i-34 in d2v_top5_list[i]:
        if i-4 in d2v_top5_list[i]:
            d2v_bool_modified.append(True)
        else:
            d2v_bool_modified.append(False)


    for i in range(0, len(bow_bool_original)):
        print(i+1)
        print(d2v_bool_original[i])
        print(d2v_bool_modified[i])
        print("\n")