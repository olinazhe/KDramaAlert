import re
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

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

def build_td_mat(documents: str) -> Tuple[np.ndarray, np.ndarray]:
  """
  Returns the term document matrix along with the terms that represent each column.
  """
  vectorizer = _build_vectorizer(5000, "english")
  X = vectorizer.fit_transform(documents).toarray()
  terms = vectorizer.get_feature_names_out()
  return X, terms
