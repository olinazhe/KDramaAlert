import numpy as np
from typing import List
from collections import defaultdict
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from nltk.stem import PorterStemmer
import json

def preprocess(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.lower().split()])

def edit_distance(a: str, b: str) -> float:
    """
    Computes Levenshtein edit distance between strings a and b.
    """
    len_a, len_b = len(a), len(b)
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[len_a][len_b]

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

    query_tokens = preprocess(query).split(" ")
    for token in query_tokens:
      if token in terms:
          index = terms.index(token)
          query_vec[index] += 1
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
      return np.zeros(n_docs)

    doc_scores = get_doc_scores(n_docs, query_tokens, terms, inv_idx, query_vec, td_mat)

    cossim = doc_scores / (np.maximum(doc_norms, 1e-10) * query_norm)
    return cossim

def get_title_sim(query: str, titles: List[str]) -> np.ndarray:
    """
    Returns the similarity between a query and each title using Jaccard similarity in the range [0, 1].
    """
    if query == "":
       return [0]*len(titles)
    query_tokens = set(query.lower().split(" "))
    result = np.zeros((len(titles),))

    for i, title in enumerate(titles):
        title_tokens = set(title.lower().split(" "))
        match_count = 0

        for q_token in query_tokens:
            for t_token in title_tokens:
                if edit_distance(q_token, t_token) <= 1:
                    match_count += 1
                    break  # prevent double-matching a single token

        union_count = len(query_tokens) + len(title_tokens) - match_count
        result[i] = match_count / union_count if union_count > 0 else 0.0

    return result

def get_social_score(ratings: List[str]) -> np.ndarray:
    result = np.zeros((len(ratings),))
    for i, rating in enumerate(ratings):
       result[i] = float(rating)/10
    return result

def svd(query, vectorizer, docs_compressed, words_compressed):
  query_tfidf = vectorizer.transform([query]).toarray()
  query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

  docs_compressed_normed = normalize(docs_compressed)
  sims = np.clip(docs_compressed_normed.dot(query_vec), 0, None)
  return sims

def get_svd_features(query, vectorizer, docs_compressed, words_compressed):
  query = "daily life"
  query_tfidf = vectorizer.transform([query]).toarray()
  print(query_tfidf.shape)
  print(words_compressed.shape)
  query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()
  print(query_vec)
  docs_compressed_normed = normalize(docs_compressed)
  word_to_index = vectorizer.vocabulary_
  index_to_word = {i:t for t,i in word_to_index.items()}
  for i in range(20):
    print("Top words in dimension", i)
    dimension_col = words_compressed[:,i].squeeze()
    asort = np.argsort(-dimension_col)
    print([index_to_word[i] for i in asort[:20]])
    print()

def get_top_latent_dims(query:str, vectorizer, words_compressed):
    
    query_tfidf = vectorizer.transform([query]).toarray()

    query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

    query_vec = np.array(query_vec)
    valid_dims = [1,2,3,4,5,6,10,12,13,17,18,19]
    query_vec = query_vec[valid_dims]
    indices = np.argsort(query_vec)[-3:][::-1]
    top_valid_dims = [valid_dims[i] for i in indices]
    word_to_index = vectorizer.vocabulary_
    index_to_word = {i:t for t,i in word_to_index.items()}
    list_of_asorts = [ np.argsort(-words_compressed[:,index].squeeze()) for index in indices]
    latent_words = [ [index_to_word[i] for i in asort[:10]] for asort in list_of_asorts]

    index_to_genre = {1: "School", 2: "Love", 3: "Family", 4: "Historical", 5:"Drama", 6: "Mystery", 10: "Medical", 12: "Friendships", 13: 'Law', 17:"College", 18:"Life", 19: "Idol"}

    genres = [index_to_genre[index] for index in top_valid_dims]
    values = query_vec[indices]
    return list(zip(genres, values, latent_words))


def get_sim(query:str, df: pd.DataFrame, td_mat: np.ndarray, inv_idx: dict, terms: List[str], doc_norms:np.ndarray, vectorizer, docs_compressed, words_compressed) -> np.ndarray:
  cossim = np.sqrt(get_cosine_similarity(query, td_mat, inv_idx, terms, doc_norms))
  title_sim = get_title_sim(query, df["name"])
  svd_sim = svd(query, vectorizer, docs_compressed, words_compressed)
  social_score = get_social_score(df["score"])
  weighted_sim = (cossim * 0.3 + title_sim * 0.35 + svd_sim * 0.25 + social_score * 0.1) * 100
  
  df['cossim'] = cossim
  df['titleSim'] = title_sim
  df['svdSim'] = svd_sim
  df['socialScore'] = social_score
  df['simScore'] = weighted_sim
  best_match_indices = np.argsort(weighted_sim)[::-1]
  best_matches = df.iloc[best_match_indices]
  best_matches = best_matches[best_matches['simScore'] > 9.2]
  return best_matches.to_json(orient="records")

def get_top_dramas_by_genre(df: pd.DataFrame, genre: str, id: str, td_matrix: np.ndarray) -> List[dict]:
    """
    Returns the top 8 dramas for a given genre.
    """

    doc_sims = np.dot(td_matrix, td_matrix[int(id)])
    sim_docs = np.argsort(-doc_sims)
    matches = df.iloc[sim_docs]
    matches = matches[matches['id'] != int(id)]
    matches = matches[matches['genres'].apply(lambda x: genre in x)].head(8)

    return matches.to_dict(orient='records')

def get_drama_details(id, df: pd.DataFrame, td_matrix: np.ndarray, docs_compressed: np.ndarray, vectorizer, words_compressed: np.ndarray):
    initial_details = df.iloc[[int(id)]].to_json(orient="records")
    initial_details = json.loads(initial_details)[0]
    similar_dramas = [[genre, get_top_dramas_by_genre(df, genre, id, td_matrix)] for genre in initial_details["genres"]]
    initial_details["similarDramas"] = similar_dramas

    valid_dims = [1,2,3,4,5,6,10,12,13,17,18,19]
    
    query_vec = docs_compressed[int(id),valid_dims]
    indices = np.argsort(query_vec)[-3:][::-1]
    top_valid_dims = [valid_dims[i] for i in indices]
    values = query_vec[indices]

    word_to_index = vectorizer.vocabulary_    
    index_to_word = {i:t for t,i in word_to_index.items()}
    list_of_asorts = [ np.argsort(-words_compressed[:,index].squeeze()) for index in top_valid_dims]
    latent_words = [ [index_to_word[i] for i in asort[:10]] for asort in list_of_asorts]
    index_to_genre = {1: "School", 2: "Love", 3: "Family", 4: "Historical", 5:"Drama", 6: "Mystery", 10: "Medical", 12: "Friendships", 13: 'Law', 17:"College", 18:"Life", 19: "Idol"}
    # print(query_vec, indices, top_valid_dims)

    genres = [index_to_genre[index] for index in top_valid_dims]
    return { "details": initial_details, "latentWords": list(zip(genres, values, latent_words))}
    
