import re
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import math
import pandas as pd
from pandas import DataFrame

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
  return kdramas_df


def strip_text(text: str, regex: str = r"\w+(?:'\w+)?") -> List[str]:
  """
  Returns a list of tokens from the text. 
  """
  
  return re.findall(regex, text.lower())

def _build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
  """
  Returns a TfidfVectorizer object with the above preprocessing properties.
  """
  
  return TfidfVectorizer(stop_words=stop_words, max_features=max_features, max_df=max_df, min_df=min_df, norm=norm)

def build_td_mat(dataframe: DataFrame) -> Tuple[np.ndarray, dict]:
  """
  Returns the term document matrix along with the terms that represent each column.
  """
  vectorizer = _build_vectorizer(5000, "english")
  doc_to_vocab = vectorizer.fit_transform(dataframe["synopsis"].to_list()).toarray()
  index_to_vocab = list(vectorizer.get_feature_names_out())
  return doc_to_vocab, index_to_vocab

def build_inverted_index(msgs: List[str]) -> dict:
    """
    Builds an inverted index .
    """
    result = defaultdict(list)
    for i, msg in enumerate(msgs):
      counts = defaultdict(int)
      tokenized_msg = msg.split(" ")
      for token in tokenized_msg:
        counts[token] += 1
      for term, count in counts.items():
        result[term].append((i, count))
    return result

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

def compute_doc_norms(index, idf, n_docs):
    """
    Precompute the euclidean norm of each document.
    """
    result = np.zeros(n_docs)
    for term in index:
      if term in idf:
        value = idf[term]
        for doc, count in index[term]:
          result[doc] += (count * value) ** 2
    return np.sqrt(result)

def drama_name_to_index(docs_df):
  """
  Returns a dictionary with keys as drama names and values as their index in the data frame.
  """
  res = {}
  for index, row in docs_df.iterrows():
    res[row["name"]] = index
  return res