from sklearn.decomposition import TruncatedSVD
import search_engine.search_engine as s_e
import numpy as np
import pickle


path = './search_engine/resources_s/'

se = s_e.SearchEngine(resources_path=path)
lsa = TruncatedSVD(n_components=100).fit(se.tfidf_matrix)
tfidf_matrix_t = lsa.transform(se.tfidf_matrix)
np.save(path + 'tfidf_matrix_t', tfidf_matrix_t)
with open(path + 'lsa.pickle', 'wb') as f:
    pickle.dump(lsa, f, pickle.HIGHEST_PROTOCOL)
