import json
import os
from helpers import preprocessing, similarity
import pandas as pd
from collections import defaultdict
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import numpy as np

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    kdramas_df = preprocessing.process_data(data)
    vectorizer, synopsis_td_mat, terms = preprocessing.build_td_mat(kdramas_df)
    inv_idx = preprocessing.build_inverted_index(synopsis_td_mat, terms)
    doc_norms = preprocessing.compute_doc_norms(synopsis_td_mat)

def svd(query, vectorizer, td_matrix):
  docs_compressed, matrix, words_compressed = svds(td_matrix, k=40)
  words_compressed = words_compressed.transpose()
  td_matrix_np = td_matrix.transpose()
  td_matrix_np = normalize(td_matrix_np)

  query_tfidf = vectorizer.transform([query]).toarray()

  query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

  docs_compressed_normed = normalize(docs_compressed)
  sims = np.clip(docs_compressed_normed.dot(query_vec), 0, None)
  print(matrix)
  return matrix
query = "business"
svd_sim = svd(query, vectorizer, synopsis_td_mat)