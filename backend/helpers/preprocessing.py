import re
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import math
import pandas as pd
from pandas import DataFrame
from nltk.stem import PorterStemmer

def preprocess(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.lower().split()])

def process_data(data):
  kdramas_df = pd.DataFrame(data)
  #tags spit differently
  convert = ['genres','network', 'tags', 'mainCast']
  for t in convert:
    if t == 'tags':
      for idx,row in kdramas_df.iterrows():
        tags = row[t]
        list_of_strings = tags.split(",, ")
        new_list = []
        for s in list_of_strings:
            new_list.append(s.strip())
        kdramas_df.at[idx,t] = new_list
    else:
      for idx,row in kdramas_df.iterrows():
        genre = row[t]
        list_of_strings = genre.split(", ")
        new_list = []
        for s in list_of_strings:
          new_list.append(s.strip())
        kdramas_df.at[idx,t] = new_list
  kdramas_df["synopsis"] = kdramas_df['synopsis'].str.replace("\\", "'")
  filtered_kdramas_df = kdramas_df[kdramas_df['synopsis'].str.strip() != '']
  filtered_kdramas_df.insert(0, "id", range(len(filtered_kdramas_df)))
  return filtered_kdramas_df


def strip_text(text: str, regex: str = r"\w+(?:'\w+)?") -> List[str]:
  """
  Returns a list of tokens from the text. 
  """
  
  return re.findall(regex, text.lower())

def _build_vectorizer(max_features, stop_words, max_df=0.95, min_df=5, norm='l2'):
  """
  Returns a TfidfVectorizer object with the above preprocessing properties.
  """
  
  return TfidfVectorizer(stop_words=stop_words, max_features=max_features, max_df=max_df, min_df=min_df, norm=norm)

def build_td_mat(df: DataFrame) -> Tuple[TfidfVectorizer, np.ndarray, list]:
  """
  Returns the term document matrix along with the terms that represent each column.
  """
  vectorizer = _build_vectorizer(7500, "english")
  preprocessed_synopses = df["synopsis"].apply(preprocess)
  synopsis_td_mat = vectorizer.fit_transform(preprocessed_synopses).toarray()
  terms = list(vectorizer.get_feature_names_out())
  return vectorizer, synopsis_td_mat, terms

def build_inverted_index(td_mat: np.ndarray, terms: List[str]) -> dict:
    """
    Builds an inverted index .
    """
    inv_idx = {}
    for term_index, term in enumerate(terms):
      col = td_mat[:, term_index]
      inv_idx[term] = []
      for doc_index, tfidf in enumerate(col):
        if tfidf != 0:
          inv_idx[term].append(doc_index)

    return inv_idx

def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.
    """
    result = {}
    for term in inv_idx:
      df = len(inv_idx[term])
      if (df >= min_df and (df / n_docs) <= max_df_ratio):
        result[term] = math.log2(n_docs / (1 + df))
    return result

def compute_doc_norms(td_mat: np.ndarray) -> np.ndarray:
    """
    Precompute the euclidean norm of each document.
    """
    return np.linalg.norm(td_mat, axis=1)

def drama_name_to_index(docs_df):
  """
  Returns a dictionary with keys as drama names and values as their index in the data frame.
  """
  res = {}
  for index, row in docs_df.iterrows():
    res[row["name"]] = index
  return res


