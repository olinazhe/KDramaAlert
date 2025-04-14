import numpy as np
from typing import List
from collections import defaultdict
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

def query_word_counts(query):
    query = query.lower().split(" ")
    query_dictionary = defaultdict(int)
    for word in query:
      query_dictionary[word] += 1
    return query_dictionary

def get_doc_scores(n_docs: int, query_tokens: List[str], terms: List[str], inv_idx: dict, query_vec: np.ndarray, td_mat: np.ndarray) -> np.ndarray:
    """
    Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.
    """
    doc_scores = np.zeros(n_docs)
    for token in query_tokens:
      if token in terms:
        for doc_index in inv_idx[token]:
           doc_scores[doc_index] += np.dot(query_vec, td_mat[doc_index])
    return doc_scores


def get_cosine_similarity(query: str, td_mat: np.ndarray, inv_idx: dict, terms: List[str], doc_norms: np.ndarray):
    """
    Search the collection of documents for the given query
    """
    n_docs, n_terms = td_mat.shape
    query_vec = np.zeros(n_terms)

    query_tokens = query.lower().split(" ")
    for token in query_tokens:
      if token in terms:
          index = terms.index(token)
          query_vec[index] += 1
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
       query_norm = 1

    doc_scores = get_doc_scores(n_docs, query_tokens, terms, inv_idx, query_vec, td_mat)

    cossim = np.divide(doc_scores, doc_norms) / query_norm
    return cossim

def get_title_sim(query: str, titles: List[str]) -> np.ndarray:
    """
    Returns the similarity between a query and each title using Jaccard similarity in the range [0, 1].
    """

    types = set(query.lower().split(" "))
    result = np.zeros((len(titles),))
    for i, title in enumerate(titles):
        title_tokens = set(title.lower().split(" "))
        result[i] = len(title_tokens.intersection(types)) / len(title_tokens)
    return result

def svd(query, vectorizer, td_matrix):
  docs_compressed, _, words_compressed = svds(td_matrix, k=40)
  words_compressed = words_compressed.transpose()
  td_matrix_np = td_matrix.transpose()
  td_matrix_np = normalize(td_matrix_np)

  query_tfidf = vectorizer.transform([query]).toarray()

  query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

  docs_compressed_normed = normalize(docs_compressed)
  sims = np.clip(docs_compressed_normed.dot(query_vec), 0, None)
  return sims

def get_sim(query:str, df: pd.DataFrame, td_mat: np.ndarray, inv_idx: dict, terms: List[str], doc_norms:np.ndarray, vectorizer) -> np.ndarray:
  cossim = get_cosine_similarity(query, td_mat, inv_idx, terms, doc_norms)
  title_sim = get_title_sim(query, df["name"])
  svd_sim = svd(query, vectorizer, td_mat)
  weighted_sim =  3 * cossim / 8 + 3 * title_sim / 8 + svd_sim / 4
  
  df['simScore'] = weighted_sim
  best_match_indices = np.argsort(weighted_sim)[::-1]
  best_matches = df.iloc[best_match_indices]
  
  return best_matches.to_json(orient="records")