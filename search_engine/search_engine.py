import numpy as np
import pickle
from os import listdir
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from scipy.sparse import save_npz, load_npz


# custom tokenizer with stemmer
class PorterTokenizer(object):
    def __init__(self):
        self.pt = PorterStemmer()

    def __call__(self, doc):
        return [self.pt.stem(t) for t in RegexpTokenizer(r'(?u)\b\w\w+\b').tokenize(doc)]


# load saved info
def load_list(path, filename):
    mylist = []
    with open(path + filename, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            mylist.append(line)
    return mylist


class SearchEngine:

    def __init__(self, resources_path):
        self.tfidf_matrix = load_npz(file=resources_path + 'tfidf_matrix.npz')
        with open(resources_path + 'tfidf_matrix_t.npy', 'rb') as f:
            self.tfidf_matrix_t = np.load(f)
        self.file_list = load_list(resources_path, 'file_list')
        with open(resources_path + 'lsa.pickle', 'rb') as f:
            self.lsa = pickle.load(f)
        with open(resources_path + 'vectorizer.pickle', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)

    def search(self, query_str):
        tfidf_search_matrix = self.tfidf_vectorizer.transform(query_str)
        sim_matrix = cosine_similarity(self.tfidf_matrix, tfidf_search_matrix)
        a = [i[0] for i in sorted(enumerate(list(sim_matrix[:, 0])), key=lambda x:x[1], reverse=True)]
        sort_m = sorted(sim_matrix[:, 0], reverse=True)
        mod_a = []
        for i in range(0, len(sort_m)):
            if sort_m[i] > 0:
                mod_a.append(a[i])
        res = [self.file_list[i] for i in mod_a]
        return res

    def lsa_search(self, query_str):
        tfidf_query_matrix = self.tfidf_vectorizer.transform(query_str)
        query = self.lsa.transform(tfidf_query_matrix)
        query = Normalizer(copy=False).transform(query)
        sim_matrix = cosine_similarity(self.tfidf_matrix_t, query)
        a = [i[0] for i in sorted(enumerate(list(sim_matrix[:, 0])), key=lambda x: x[1], reverse=True)]
        sort_m = sorted(sim_matrix[:, 0], reverse=True)
        mod_a = []
        for i in range(0, len(sort_m)):
            if sort_m[i] > 0:
                mod_a.append(a[i])
        res = [self.file_list[i] for i in mod_a]
        return res


# read each file into separate string

def read_docs(art_path):
    file_list = [name for name in listdir(art_path)]
    doc_list = []
    for file in file_list:
        with open(art_path + file, 'r') as myfile:
            doc_list.append(myfile.read())
    return file_list, doc_list


# vectorize strings using tf-idf vectorizer

def tfidf_transform(doc_list):
    tfidf_vectorizer = TfidfVectorizer(min_df=1,
                                       stop_words='english',
                                       tokenizer=PorterTokenizer(),
                                       smooth_idf=True).fit(doc_list)
    tfidf_trans_matrix = tfidf_vectorizer.transform(doc_list)
    return tfidf_vectorizer, tfidf_trans_matrix


def lsa_learn(tfidf_matrix, n_comp):
    lsa = TruncatedSVD(n_components=n_comp).fit(tfidf_matrix)
    tfidf_matrix_t = lsa.transform(tfidf_matrix)
    return lsa, tfidf_matrix_t


def save_list(path, filename, l):
    with open(path + filename, 'w+') as f:
        for item in l:
            f.write("%s\n" % item)


class SearchIndexer:
    def __init__(self, art_path):
        self.art_path = art_path
        self.file_list, self.doc_list = read_docs(self.art_path)
        self.tfidf_vectorizer, self.tfidf_matrix = tfidf_transform(self.doc_list)
        self.lsa, self.tfidf_matrix_t = lsa_learn(self.tfidf_matrix, 100)

    def save(self, path):
        save_npz(file=path + 'tfidf_matrix', matrix=self.tfidf_matrix)
        np.save(path + 'tfidf_matrix_t', self.tfidf_matrix_t)
        save_list(path, 'file_list', self.file_list)
        with open(path + 'vectorizer.pickle', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f, pickle.HIGHEST_PROTOCOL)
        with open(path + 'lsa.pickle', 'wb') as f:
            pickle.dump(self.lsa, f, pickle.HIGHEST_PROTOCOL)
